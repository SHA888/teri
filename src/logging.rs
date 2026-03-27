use crate::error::Result;
use tracing_subscriber::{EnvFilter, fmt};

pub fn init_logging(level: &str) -> Result<()> {
    let env_filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(level))
        .unwrap_or_else(|_| EnvFilter::new("info"));

    fmt().with_env_filter(env_filter).with_target(true).with_level(true).init();

    Ok(())
}
