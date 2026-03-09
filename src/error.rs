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

// Common conversions
impl From<reqwest::Error> for TeriError {
    fn from(err: reqwest::Error) -> Self {
        TeriError::Http(err.to_string())
    }
}

impl From<redb::Error> for TeriError {
    fn from(err: redb::Error) -> Self {
        TeriError::Database(err.to_string())
    }
}

impl From<bincode::Error> for TeriError {
    fn from(err: bincode::Error) -> Self {
        TeriError::Serialization(err.to_string())
    }
}

impl From<config::ConfigError> for TeriError {
    fn from(err: config::ConfigError) -> Self {
        TeriError::Config(err.to_string())
    }
}

// Lightweight context helper
pub trait ResultExt<T> {
    fn with_context<F: FnOnce() -> String>(self, ctx: F) -> Result<T>;
}

impl<T, E> ResultExt<T> for std::result::Result<T, E>
where
    E: Into<TeriError>,
{
    fn with_context<F: FnOnce() -> String>(self, ctx: F) -> Result<T> {
        self.map_err(|e| TeriError::Unknown(format!("{}: {}", ctx(), e.into())))
    }
}
