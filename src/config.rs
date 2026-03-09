use crate::error::{Result, TeriError};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub llm: LlmConfig,
    pub sim: SimConfig,
    pub persistence: PersistenceConfig,
    pub api: ApiConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub embed_model: String,
    pub timeout_secs: u64,
    pub max_retries: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimConfig {
    pub default_agent_count: usize,
    pub max_ticks: u32,
    pub parallelism: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    pub memory_db_path: String,
    pub graph_db_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub bind_addr: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        dotenvy::dotenv().ok();

        let config = Self {
            llm: LlmConfig {
                base_url: std::env::var("LLM_BASE_URL")
                    .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
                api_key: std::env::var("LLM_API_KEY")
                    .map_err(|_| TeriError::Config("LLM_API_KEY not set".to_string()))?,
                model: std::env::var("LLM_MODEL")
                    .unwrap_or_else(|_| "gpt-4o".to_string()),
                embed_model: std::env::var("EMBED_MODEL")
                    .unwrap_or_else(|_| "text-embedding-3-small".to_string()),
                timeout_secs: std::env::var("LLM_TIMEOUT_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(30),
                max_retries: std::env::var("LLM_MAX_RETRIES")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(3),
            },
            sim: SimConfig {
                default_agent_count: std::env::var("DEFAULT_AGENT_COUNT")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(100),
                max_ticks: std::env::var("SIM_MAX_TICKS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(50),
                parallelism: std::env::var("SIM_PARALLELISM")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(8),
            },
            persistence: PersistenceConfig {
                memory_db_path: std::env::var("MEMORY_DB_PATH")
                    .unwrap_or_else(|_| "./data/memory.db".to_string()),
                graph_db_path: std::env::var("GRAPH_DB_PATH")
                    .unwrap_or_else(|_| "./data/graph".to_string()),
            },
            api: ApiConfig {
                bind_addr: std::env::var("BIND_ADDR")
                    .unwrap_or_else(|_| "0.0.0.0:8080".to_string()),
            },
            logging: LoggingConfig {
                level: std::env::var("RUST_LOG")
                    .unwrap_or_else(|_| "teri=debug,tower_http=info".to_string()),
            },
        };

        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.llm.api_key.is_empty() {
            return Err(TeriError::Config("LLM_API_KEY cannot be empty".to_string()));
        }

        if self.sim.default_agent_count == 0 {
            return Err(TeriError::Config("DEFAULT_AGENT_COUNT must be > 0".to_string()));
        }

        if self.sim.max_ticks == 0 {
            return Err(TeriError::Config("SIM_MAX_TICKS must be > 0".to_string()));
        }

        if self.sim.parallelism == 0 {
            return Err(TeriError::Config("SIM_PARALLELISM must be > 0".to_string()));
        }

        Ok(())
    }
}
