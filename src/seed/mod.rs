use crate::error::{Result, TeriError};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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
        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| TeriError::Seed(format!("Failed to read file: {e}")))?;

        let metadata = Self::extract_file_metadata(path)?;

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
        let text = resp
            .text()
            .await
            .map_err(|e| TeriError::Seed(format!("Failed to read response: {e}")))?;

        let mut metadata = HashMap::new();
        metadata.insert("source_url".to_string(), url.to_string());

        Ok(SeedDocument {
            id: Uuid::new_v4(),
            raw_text: text,
            metadata,
            created_at: Utc::now(),
        })
    }

    fn extract_file_metadata(path: &str) -> Result<HashMap<String, String>> {
        let mut metadata = HashMap::new();
        metadata.insert("source_path".to_string(), path.to_string());

        if let Some(filename) = std::path::Path::new(path).file_name() {
            if let Some(name_str) = filename.to_str() {
                metadata.insert("filename".to_string(), name_str.to_string());
            }
        }

        Ok(metadata)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_seed_ingestor_from_file() {
        let test_file = "/tmp/test_seed.txt";
        let test_content = "This is a test seed document";

        tokio::fs::write(test_file, test_content)
            .await
            .expect("Failed to write test file");

        let doc = SeedIngestor::from_file(test_file)
            .await
            .expect("Failed to ingest seed");

        assert_eq!(doc.raw_text, test_content);
        assert!(doc.metadata.contains_key("source_path"));
        assert!(doc.metadata.contains_key("filename"));

        let _ = tokio::fs::remove_file(test_file).await;
    }
}
