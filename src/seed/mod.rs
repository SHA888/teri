use crate::error::{Result, TeriError};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeedDocument {
    pub id: Uuid,
    pub raw_text: String,
    pub metadata: HashMap<String, String>,
    pub created_at: chrono::DateTime<Utc>,
}

pub struct SeedIngestor;

impl SeedIngestor {
    pub async fn from_file(path: &str) -> Result<SeedDocument> {
        let path_obj = Path::new(path);
        let extension = path_obj
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        let (content, mut metadata) = match extension.as_str() {
            "txt" => Self::read_plain_text(path).await?,
            "pdf" => Self::read_pdf(path).await?,
            "json" => Self::read_json(path).await?,
            _ => Self::read_plain_text(path).await?,
        };

        metadata.insert("source_path".to_string(), path.to_string());
        if let Some(filename) = path_obj.file_name() {
            if let Some(name_str) = filename.to_str() {
                metadata.insert("filename".to_string(), name_str.to_string());
            }
        }
        metadata.insert("file_format".to_string(), extension);

        Ok(SeedDocument {
            id: Uuid::new_v4(),
            raw_text: content,
            metadata,
            created_at: Utc::now(),
        })
    }

    pub async fn from_url(url: &str) -> Result<SeedDocument> {
        let resp = reqwest::get(url)
            .await
            .map_err(|e| TeriError::Seed(format!("Failed to fetch URL: {e}")))?;

        if !resp.status().is_success() {
            return Err(TeriError::Seed(format!(
                "Failed to fetch URL: received status {}",
                resp.status()
            )));
        }

        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_lowercase();

        let html = resp
            .text()
            .await
            .map_err(|e| TeriError::Seed(format!("Failed to read response: {e}")))?;

        let (content, mut metadata) = Self::extract_web_content(&html)?;

        metadata.insert("source_url".to_string(), url.to_string());
        metadata.insert("content_type".to_string(), content_type);

        Ok(SeedDocument {
            id: Uuid::new_v4(),
            raw_text: content,
            metadata,
            created_at: Utc::now(),
        })
    }

    async fn read_plain_text(path: &str) -> Result<(String, HashMap<String, String>)> {
        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| TeriError::Seed(format!("Failed to read text file: {e}")))?;

        let metadata = Self::extract_basic_file_metadata(path)?;

        Ok((content, metadata))
    }

    async fn read_pdf(path: &str) -> Result<(String, HashMap<String, String>)> {
        use pdfium_render::prelude::*;

        let bindings = Pdfium::bind_to_system_library()
            .map_err(|e| TeriError::Seed(format!("Failed to load pdfium library: {e:?}")))?;
        let pdfium = Pdfium::new(bindings);

        let document = pdfium
            .load_pdf_from_file(path, None)
            .map_err(|e| TeriError::Seed(format!("Failed to parse PDF: {e:?}")))?;

        let mut text_content = String::new();
        let mut metadata = HashMap::new();

        let page_count = document.pages().len();
        metadata.insert("page_count".to_string(), page_count.to_string());

        for page in document.pages().iter() {
            match page.text() {
                Ok(page_text) => {
                    text_content.push_str(&page_text.all());
                    text_content.push('\n');
                }
                Err(_) => continue,
            }
        }

        metadata.extend(Self::extract_basic_file_metadata(path)?);

        Ok((text_content, metadata))
    }

    async fn read_json(path: &str) -> Result<(String, HashMap<String, String>)> {
        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| TeriError::Seed(format!("Failed to read JSON file: {e}")))?;

        let json_value: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| TeriError::Seed(format!("Failed to parse JSON: {e}")))?;

        let text_content = Self::json_to_text(&json_value);
        let mut metadata = Self::extract_basic_file_metadata(path)?;
        metadata.insert(
            "json_structure".to_string(),
            Self::describe_json_structure(&json_value),
        );

        Ok((text_content, metadata))
    }

    fn json_to_text(value: &serde_json::Value) -> String {
        match value {
            serde_json::Value::Object(map) => {
                let mut result = String::new();
                for (key, val) in map.iter() {
                    result.push_str(&format!("{}: {}\n", key, Self::json_to_text(val)));
                }
                result
            }
            serde_json::Value::Array(arr) => {
                let mut result = String::new();
                for (idx, item) in arr.iter().enumerate() {
                    result.push_str(&format!("[{}] {}\n", idx, Self::json_to_text(item)));
                }
                result
            }
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::Bool(b) => b.to_string(),
            serde_json::Value::Null => "null".to_string(),
        }
    }

    fn describe_json_structure(value: &serde_json::Value) -> String {
        match value {
            serde_json::Value::Object(map) => {
                let keys: Vec<_> = map.keys().cloned().collect();
                format!("object with keys: {}", keys.join(", "))
            }
            serde_json::Value::Array(arr) => {
                format!("array with {} items", arr.len())
            }
            serde_json::Value::String(_) => "string".to_string(),
            serde_json::Value::Number(_) => "number".to_string(),
            serde_json::Value::Bool(_) => "boolean".to_string(),
            serde_json::Value::Null => "null".to_string(),
        }
    }

    fn extract_web_content(html: &str) -> Result<(String, HashMap<String, String>)> {
        use scraper::{Html, Selector};

        let document = Html::parse_document(html);
        let mut metadata = HashMap::new();

        let title_selector = Selector::parse("title")
            .map_err(|_| TeriError::Seed("Failed to parse title selector".to_string()))?;
        if let Some(title_elem) = document.select(&title_selector).next() {
            if let Some(title_text) = title_elem.inner_html().lines().next() {
                metadata.insert("title".to_string(), title_text.trim().to_string());
            }
        }

        let meta_desc_selector = Selector::parse("meta[name=\"description\"]")
            .map_err(|_| TeriError::Seed("Failed to parse meta selector".to_string()))?;
        if let Some(meta_elem) = document.select(&meta_desc_selector).next() {
            if let Some(content) = meta_elem.value().attr("content") {
                metadata.insert("description".to_string(), content.to_string());
            }
        }

        let meta_author_selector = Selector::parse("meta[name=\"author\"]")
            .map_err(|_| TeriError::Seed("Failed to parse author selector".to_string()))?;
        if let Some(meta_elem) = document.select(&meta_author_selector).next() {
            if let Some(content) = meta_elem.value().attr("content") {
                metadata.insert("author".to_string(), content.to_string());
            }
        }

        let body_selector = Selector::parse("body")
            .map_err(|_| TeriError::Seed("Failed to parse body selector".to_string()))?;

        let text_content = if let Some(body) = document.select(&body_selector).next() {
            let script_style_selector = Selector::parse("script, style").map_err(|_| {
                TeriError::Seed("Failed to parse script/style selector".to_string())
            })?;

            let mut body_html = body.inner_html();
            for elem in body.select(&script_style_selector) {
                body_html = body_html.replace(&elem.inner_html(), "");
            }

            let body_doc = Html::parse_fragment(&body_html);
            body_doc
                .root_element()
                .inner_html()
                .lines()
                .map(|line| line.trim())
                .filter(|line| !line.is_empty())
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            html.to_string()
        };

        Ok((text_content, metadata))
    }

    fn extract_basic_file_metadata(path: &str) -> Result<HashMap<String, String>> {
        let mut metadata = HashMap::new();

        if let Ok(file_metadata) = std::fs::metadata(path) {
            if let Ok(size) = file_metadata.len().to_string().parse::<u64>() {
                metadata.insert("file_size_bytes".to_string(), size.to_string());
            }

            if let Ok(modified) = file_metadata.modified() {
                if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
                    let timestamp =
                        chrono::DateTime::<Utc>::from(std::time::SystemTime::UNIX_EPOCH + duration);
                    metadata.insert("modified_date".to_string(), timestamp.to_rfc3339());
                }
            }
        }

        Ok(metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;
    use std::fs;

    #[tokio::test]
    async fn test_plain_text_file_format() {
        let test_file = "/tmp/test_plain_text.txt";
        let test_content =
            "This is a test seed document\nWith multiple lines\nFor testing purposes";

        tokio::fs::write(test_file, test_content)
            .await
            .expect("Failed to write test file");

        let doc = SeedIngestor::from_file(test_file)
            .await
            .expect("Failed to ingest seed");

        assert_eq!(doc.raw_text, test_content);
        assert_eq!(doc.metadata.get("file_format").unwrap(), "txt");
        assert!(doc.metadata.contains_key("source_path"));
        assert!(doc.metadata.contains_key("filename"));
        assert!(doc.metadata.contains_key("file_size_bytes"));
        assert!(doc.metadata.contains_key("modified_date"));

        let _ = tokio::fs::remove_file(test_file).await;
    }

    #[tokio::test]
    async fn test_json_file_format() {
        let test_file = "/tmp/test_data.json";
        let test_content = r#"{"name": "Test", "value": 42, "items": ["a", "b", "c"]}"#;

        tokio::fs::write(test_file, test_content)
            .await
            .expect("Failed to write test file");

        let doc = SeedIngestor::from_file(test_file)
            .await
            .expect("Failed to ingest seed");

        assert_eq!(doc.metadata.get("file_format").unwrap(), "json");
        assert!(doc.metadata.contains_key("json_structure"));
        assert!(doc.raw_text.contains("name"));
        assert!(doc.raw_text.contains("Test"));
        assert!(doc.raw_text.contains("value"));
        assert!(doc.raw_text.contains("42"));

        let _ = tokio::fs::remove_file(test_file).await;
    }

    #[tokio::test]
    async fn test_json_to_text_conversion() {
        let json_obj = serde_json::json!({
            "title": "Test Document",
            "author": "Test Author",
            "content": "This is test content"
        });

        let text = SeedIngestor::json_to_text(&json_obj);
        assert!(text.contains("title"));
        assert!(text.contains("Test Document"));
        assert!(text.contains("author"));
        assert!(text.contains("Test Author"));
    }

    #[tokio::test]
    async fn test_json_structure_description() {
        let json_obj = serde_json::json!({
            "items": [1, 2, 3],
            "metadata": {"key": "value"}
        });

        let description = SeedIngestor::describe_json_structure(&json_obj);
        assert!(description.contains("object with keys"));
        assert!(description.contains("items"));
        assert!(description.contains("metadata"));
    }

    #[tokio::test]
    async fn test_web_content_extraction() {
        let html = r#"
            <html>
                <head>
                    <title>Test Page</title>
                    <meta name="description" content="Test Description">
                    <meta name="author" content="Test Author">
                </head>
                <body>
                    <h1>Main Content</h1>
                    <p>This is the main content of the page.</p>
                    <script>console.log('hidden');</script>
                </body>
            </html>
        "#;

        let (content, metadata) =
            SeedIngestor::extract_web_content(html).expect("Failed to extract web content");

        assert!(content.contains("Main Content"));
        assert!(content.contains("main content"));
        assert!(!content.contains("console.log"));
        assert_eq!(metadata.get("title").unwrap(), "Test Page");
        assert_eq!(metadata.get("description").unwrap(), "Test Description");
        assert_eq!(metadata.get("author").unwrap(), "Test Author");
    }

    #[tokio::test]
    async fn test_basic_file_metadata_extraction() {
        let test_file = "/tmp/test_metadata.txt";
        let test_content = "Test content for metadata extraction";

        tokio::fs::write(test_file, test_content)
            .await
            .expect("Failed to write test file");

        let metadata = SeedIngestor::extract_basic_file_metadata(test_file)
            .expect("Failed to extract metadata");

        assert!(metadata.contains_key("file_size_bytes"));
        assert!(metadata.contains_key("modified_date"));

        let file_size: u64 = metadata
            .get("file_size_bytes")
            .unwrap()
            .parse()
            .expect("Failed to parse file size");
        assert!(file_size > 0);

        let _ = tokio::fs::remove_file(test_file).await;
    }

    #[tokio::test]
    async fn test_unknown_file_format_defaults_to_plain_text() {
        let test_file = "/tmp/test_unknown.xyz";
        let test_content = "Unknown format content";

        tokio::fs::write(test_file, test_content)
            .await
            .expect("Failed to write test file");

        let doc = SeedIngestor::from_file(test_file)
            .await
            .expect("Failed to ingest seed");

        assert_eq!(doc.raw_text, test_content);
        assert_eq!(doc.metadata.get("file_format").unwrap(), "xyz");

        let _ = tokio::fs::remove_file(test_file).await;
    }

    #[tokio::test]
    async fn test_seed_document_creation() {
        let test_file = "/tmp/test_seed_doc.txt";
        let test_content = "Test seed document";

        tokio::fs::write(test_file, test_content)
            .await
            .expect("Failed to write test file");

        let doc = SeedIngestor::from_file(test_file)
            .await
            .expect("Failed to ingest seed");

        assert!(!doc.id.to_string().is_empty());
        assert_eq!(doc.raw_text, test_content);
        assert!(!doc.metadata.is_empty());
        assert!(doc.created_at <= Utc::now());

        let _ = tokio::fs::remove_file(test_file).await;
    }

    #[tokio::test]
    async fn test_missing_file_returns_error() {
        let res = SeedIngestor::from_file("/tmp/does_not_exist_123.txt").await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn test_malformed_json_returns_error() {
        let test_file = "/tmp/test_bad.json";
        let bad_content = "{ invalid json }";

        tokio::fs::write(test_file, bad_content)
            .await
            .expect("Failed to write test file");

        let res = SeedIngestor::from_file(test_file).await;
        assert!(res.is_err());

        let _ = tokio::fs::remove_file(test_file).await;
    }

    #[tokio::test]
    async fn test_invalid_pdf_returns_error() {
        let test_file = "/tmp/test_bad.pdf";
        tokio::fs::write(test_file, "not a real pdf")
            .await
            .expect("Failed to write test file");

        let res = SeedIngestor::from_file(test_file).await;
        assert!(res.is_err());

        let _ = tokio::fs::remove_file(test_file).await;
    }

    #[tokio::test]
    async fn test_url_non_success_status_errors() {
        let server = MockServer::start_async().await;
        let _mock = server
            .mock_async(|when, then| {
                when.method(GET).path("/fail");
                then.status(500).body("error");
            })
            .await;

        let res = SeedIngestor::from_url(&format!("{}/fail", server.base_url())).await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn test_integration_examples() {
        // Plain text
        let txt = SeedIngestor::from_file("examples/seed.txt")
            .await
            .expect("should ingest txt");
        assert!(txt.raw_text.contains("sample seed text"));

        // JSON
        let json = SeedIngestor::from_file("examples/news.json")
            .await
            .expect("should ingest json");
        assert!(json.raw_text.contains("headline"));
        assert!(json.metadata.contains_key("json_structure"));

        // PDF (skip assertions if pdfium system library is unavailable)
        match SeedIngestor::from_file("examples/policy.pdf").await {
            Ok(pdf) => {
                assert!(pdf.raw_text.to_lowercase().contains("hello"));
                assert_eq!(pdf.metadata.get("page_count"), Some(&"1".to_string()));
            }
            Err(e) => {
                let msg = format!("{e}");
                if !msg.contains("Failed to load pdfium library") {
                    panic!("unexpected pdf ingestion failure: {msg}");
                }
            }
        }

        // Web (served from local mock)
        let html_body = fs::read_to_string("examples/web_article.html").expect("read html fixture");
        let server = MockServer::start_async().await;
        let _mock = server
            .mock_async(|when, then| {
                when.method(GET).path("/article");
                then.status(200)
                    .header("content-type", "text/html")
                    .body(html_body.clone());
            })
            .await;

        let web = SeedIngestor::from_url(&format!("{}/article", server.base_url()))
            .await
            .expect("should ingest web html");
        assert!(web.raw_text.contains("Integration Test Article"));
        assert_eq!(
            web.metadata.get("title"),
            Some(&"Example Article".to_string())
        );
        assert_eq!(
            web.metadata.get("author"),
            Some(&"Integration Bot".to_string())
        );
    }
}
