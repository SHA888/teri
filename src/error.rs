use thiserror::Error;

#[derive(Error, Debug)]
pub enum TeriError {
    #[error("Seed error: {0}")]
    Seed(String),

    #[error("Graph error: {0}")]
    Graph(String),

    #[error("Agent error: {0}")]
    Agent(String),

    #[error("Simulation error: {0}")]
    Sim(String),

    #[error("Memory error: {0}")]
    Memory(String),

    #[error("Report error: {0}")]
    Report(String),

    #[error("API error: {0}")]
    Api(String),

    #[error("LLM error: {0}")]
    Llm(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    Http(String),

    #[error("Database error: {0}")]
    Database(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

pub type Result<T> = std::result::Result<T, TeriError>;
