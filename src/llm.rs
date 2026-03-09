use crate::config::LlmConfig;
use crate::error::{Result, TeriError};
use async_stream::try_stream;
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde::de::DeserializeOwned;
use std::pin::Pin;

/// Core LLM client trait - completely provider-agnostic.
/// This trait makes NO assumptions about the underlying provider.
#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn complete(&self, prompt: &str) -> Result<String>;
    async fn complete_json<T: DeserializeOwned>(&self, prompt: &str) -> Result<T>;
    async fn stream(&self, prompt: &str) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>>;
}

// ============================================================================
// Provider Adapters
// ============================================================================
// Each adapter implements LlmClient for a specific provider's API.
// Users can choose which adapter to use, or implement their own.

/// Adapter for providers using OpenAI's chat completions API format.
/// Examples: OpenAI, Ollama, LM Studio, vLLM, Together AI, Groq
pub struct OpenAiAdapter {
    base_url: String,
    api_key: String,
    model: String,
    client: reqwest::Client,
    timeout_secs: u64,
    max_retries: u32,
}

impl OpenAiAdapter {
    pub fn new(config: &LlmConfig) -> Self {
        let client = reqwest::Client::new();
        Self {
            base_url: config.base_url.clone(),
            api_key: config.api_key.clone(),
            model: config.model.clone(),
            client,
            timeout_secs: config.timeout_secs,
            max_retries: config.max_retries,
        }
    }

    async fn call_api(&self, payload: serde_json::Value) -> Result<serde_json::Value> {
        let url = format!("{}/chat/completions", self.base_url);
        let mut retries = 0;

        loop {
            let response = self
                .client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .timeout(std::time::Duration::from_secs(self.timeout_secs))
                .json(&payload)
                .send()
                .await;

            match response {
                Ok(resp) => {
                    if resp.status().is_success() {
                        return resp
                            .json()
                            .await
                            .map_err(|e| TeriError::Http(e.to_string()));
                    } else if resp.status().is_server_error() && retries < self.max_retries {
                        retries += 1;
                        tokio::time::sleep(std::time::Duration::from_secs(2_u64.pow(retries))).await;
                        continue;
                    } else {
                        let status = resp.status();
                        let body = resp.text().await.unwrap_or_default();
                        return Err(TeriError::Http(format!(
                            "HTTP {}: {}",
                            status, body
                        )));
                    }
                }
                Err(e) if retries < self.max_retries && e.is_timeout() => {
                    retries += 1;
                    tokio::time::sleep(std::time::Duration::from_secs(2_u64.pow(retries))).await;
                    continue;
                }
                Err(e) => return Err(TeriError::Http(e.to_string())),
            }
        }
    }
}

#[async_trait]
impl LlmClient for OpenAiAdapter {
    async fn complete(&self, prompt: &str) -> Result<String> {
        let payload = serde_json::json!({
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
        });

        let response = self.call_api(payload).await?;

        response
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| TeriError::Llm("Invalid response format".to_string()))
    }

    async fn complete_json<T: DeserializeOwned>(&self, prompt: &str) -> Result<T> {
        let payload = serde_json::json!({
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.0,
            "response_format": {
                "type": "json_object"
            }
        });

        let response = self.call_api(payload).await?;

        let content = response
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .ok_or_else(|| TeriError::Llm("Invalid response format".to_string()))?;

        serde_json::from_str(content)
            .map_err(|e| TeriError::Llm(format!("Failed to parse JSON response: {}", e)))
    }

    async fn stream(&self, prompt: &str) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
        let payload = serde_json::json!({
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "stream": true,
        });

        let url = format!("{}/chat/completions", self.base_url);
        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .json(&payload)
            .send()
            .await
            .map_err(|e| TeriError::Http(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(TeriError::Http(format!(
                "HTTP {}: {}",
                resp.status(),
                resp.text().await.unwrap_or_default()
            )));
        }

        let mut byte_stream = resp.bytes_stream();
        let sse_stream = try_stream! {
            let mut buffer = String::new();
            while let Some(chunk) = byte_stream.next().await {
                let chunk = chunk.map_err(|e| TeriError::Http(e.to_string()))?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(idx) = buffer.find('\n') {
                    let line = buffer[..idx].trim_end_matches('\r').to_string();
                    buffer = buffer[idx + 1..].to_string();

                    if line.is_empty() || !line.starts_with("data: ") {
                        continue;
                    }

                    let data = line.trim_start_matches("data: ").trim();
                    if data == "[DONE]" {
                        return;
                    }

                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                        if let Some(content) = json
                            .get("choices")
                            .and_then(|c| c.get(0))
                            .and_then(|c| c.get("delta"))
                            .and_then(|d| d.get("content"))
                            .and_then(|c| c.as_str()) {
                                yield Ok(content.to_string());
                        }
                    }
                }
            }
        };

        Ok(Box::pin(sse_stream))
    }
}

// ============================================================================
// Anthropic Claude Adapter
// ============================================================================

/// Adapter for Anthropic Claude API (non-OpenAI-compatible).
/// Uses Anthropic's Messages API format.
pub struct AnthropicAdapter {
    api_key: String,
    model: String,
    client: reqwest::Client,
    timeout_secs: u64,
    max_retries: u32,
}

impl AnthropicAdapter {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            client: reqwest::Client::new(),
            timeout_secs: 30,
            max_retries: 3,
        }
    }

    async fn call_api(&self, payload: serde_json::Value) -> Result<serde_json::Value> {
        let url = "https://api.anthropic.com/v1/messages";
        let mut retries = 0;

        loop {
            let response = self
                .client
                .post(url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", "2023-06-01")
                .header("Content-Type", "application/json")
                .timeout(std::time::Duration::from_secs(self.timeout_secs))
                .json(&payload)
                .send()
                .await;

            match response {
                Ok(resp) => {
                    if resp.status().is_success() {
                        return resp
                            .json()
                            .await
                            .map_err(|e| TeriError::Http(e.to_string()));
                    } else if resp.status().is_server_error() && retries < self.max_retries {
                        retries += 1;
                        tokio::time::sleep(std::time::Duration::from_secs(2_u64.pow(retries))).await;
                        continue;
                    } else {
                        let status = resp.status();
                        let body = resp.text().await.unwrap_or_default();
                        return Err(TeriError::Http(format!(
                            "HTTP {}: {}",
                            status, body
                        )));
                    }
                }
                Err(e) if retries < self.max_retries && e.is_timeout() => {
                    retries += 1;
                    tokio::time::sleep(std::time::Duration::from_secs(2_u64.pow(retries))).await;
                    continue;
                }
                Err(e) => return Err(TeriError::Http(e.to_string())),
            }
        }
    }
}

#[async_trait]
impl LlmClient for AnthropicAdapter {
    async fn complete(&self, prompt: &str) -> Result<String> {
        let payload = serde_json::json!({
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 4096,
        });

        let response = self.call_api(payload).await?;

        response
            .get("content")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| TeriError::Llm("Invalid response format".to_string()))
    }

    async fn complete_json<T: DeserializeOwned>(&self, prompt: &str) -> Result<T> {
        let json_prompt = format!("{}\n\nRespond with valid JSON only.", prompt);
        let response = self.complete(&json_prompt).await?;
        
        serde_json::from_str(&response)
            .map_err(|e| TeriError::Llm(format!("Failed to parse JSON response: {}", e)))
    }

    async fn stream(&self, prompt: &str) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
        // Simplified streaming - for now just return complete response as single chunk
        // TODO: Implement proper SSE streaming with Anthropic's streaming API
        let response = self.complete(prompt).await?;
        let stream = futures::stream::once(async move { Ok(response) });
        Ok(Box::pin(stream))
    }
}

// ============================================================================
// Google Gemini Adapter
// ============================================================================

/// Adapter for Google Gemini API (non-OpenAI-compatible).
/// Uses Google's generateContent API format.
pub struct GeminiAdapter {
    api_key: String,
    model: String,
    client: reqwest::Client,
    timeout_secs: u64,
    max_retries: u32,
}

impl GeminiAdapter {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            client: reqwest::Client::new(),
            timeout_secs: 30,
            max_retries: 3,
        }
    }

    async fn call_api(&self, payload: serde_json::Value) -> Result<serde_json::Value> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model, self.api_key
        );
        let mut retries = 0;

        loop {
            let response = self
                .client
                .post(&url)
                .header("Content-Type", "application/json")
                .timeout(std::time::Duration::from_secs(self.timeout_secs))
                .json(&payload)
                .send()
                .await;

            match response {
                Ok(resp) => {
                    if resp.status().is_success() {
                        return resp
                            .json()
                            .await
                            .map_err(|e| TeriError::Http(e.to_string()));
                    } else if resp.status().is_server_error() && retries < self.max_retries {
                        retries += 1;
                        tokio::time::sleep(std::time::Duration::from_secs(2_u64.pow(retries))).await;
                        continue;
                    } else {
                        let status = resp.status();
                        let body = resp.text().await.unwrap_or_default();
                        return Err(TeriError::Http(format!(
                            "HTTP {}: {}",
                            status, body
                        )));
                    }
                }
                Err(e) if retries < self.max_retries && e.is_timeout() => {
                    retries += 1;
                    tokio::time::sleep(std::time::Duration::from_secs(2_u64.pow(retries))).await;
                    continue;
                }
                Err(e) => return Err(TeriError::Http(e.to_string())),
            }
        }
    }
}

#[async_trait]
impl LlmClient for GeminiAdapter {
    async fn complete(&self, prompt: &str) -> Result<String> {
        let payload = serde_json::json!({
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        });

        let response = self.call_api(payload).await?;

        response
            .get("candidates")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("content"))
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.get(0))
            .and_then(|p| p.get("text"))
            .and_then(|t| t.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| TeriError::Llm("Invalid response format".to_string()))
    }

    async fn complete_json<T: DeserializeOwned>(&self, prompt: &str) -> Result<T> {
        let json_prompt = format!("{}\n\nRespond with valid JSON only.", prompt);
        let response = self.complete(&json_prompt).await?;
        
        serde_json::from_str(&response)
            .map_err(|e| TeriError::Llm(format!("Failed to parse JSON response: {}", e)))
    }

    async fn stream(&self, prompt: &str) -> Result<Pin<Box<dyn Stream<Item = Result<String>> + Send>>> {
        // Simplified streaming - for now just return complete response as single chunk
        // TODO: Implement proper streaming with Gemini's streamGenerateContent API
        let response = self.complete(prompt).await?;
        let stream = futures::stream::once(async move { Ok(response) });
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_adapter_creation() {
        let config = LlmConfig {
            base_url: "https://api.openai.com/v1".to_string(),
            api_key: "test-key".to_string(),
            model: "gpt-4o".to_string(),
            embed_model: "text-embedding-3-small".to_string(),
            timeout_secs: 30,
            max_retries: 3,
        };

        let _client = OpenAiAdapter::new(&config);
    }
}
