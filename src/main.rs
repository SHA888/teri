use clap::{Parser, Subcommand};
use teri::{Config, Result, TeriError};

#[derive(Parser)]
#[command(name = "teri", version, about = "Swarm Intelligence Prediction Engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Ingest seed material and launch a simulation world
    Run {
        #[arg(short, long)]
        seed: String,
        #[arg(short, long)]
        query: String,
        #[arg(short, long, default_value_t = 100)]
        agents: usize,
    },
    /// Start the REST API server
    Serve {
        #[arg(short, long, default_value = "0.0.0.0:8080")]
        addr: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    let config = Config::from_env()?;
    tracing_subscriber::fmt().with_env_filter(&config.logging.level).init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Run { seed, query, agents } => {
            tracing::info!("Starting simulation: seed={seed}, agents={agents}");
            tracing::info!("Query: {query}");
            tracing::info!("Configuration loaded successfully");
            Err(TeriError::Unknown("Pipeline not yet implemented".to_string()))
        }
        Commands::Serve { addr } => {
            tracing::info!("Starting API server on {addr}");
            Err(TeriError::Unknown("API server not yet implemented".to_string()))
        }
    }
}
