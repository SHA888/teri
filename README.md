# Teri

> **Rust-native Swarm Intelligence Prediction Engine**  
> A ground-up rewrite of [MiroFish](https://github.com/666ghj/MiroFish) — designed for performance, type safety, and deployment simplicity.

---

## What is Teri?

Teri turns seed materials (news articles, policy drafts, financial signals, novels) into a **high-fidelity parallel digital world** populated by thousands of independent agents. Each agent carries its own persona, long-term memory, and behavioural logic. The swarm self-organises, and you observe — or intervene — from a God's-eye view.

**Input** → seed file + natural-language prediction query  
**Output** → structured prediction report + interactive living simulation world

---

## Why Rust?

| Concern | Python (MiroFish) | Teri (Rust) |
|---|---|---|
| Agent parallelism | GIL-limited threads | `rayon` true parallelism |
| Memory per agent | ~MB overhead | Controlled, stack-friendly |
| Deployment | Docker + venv | Single static binary |
| Type safety | Runtime errors | Compile-time guarantees |
| Async LLM calls | asyncio | `tokio` native |

---

## Quick Start

```bash
# Copy and fill environment config
cp .env.example .env

# Run a simulation
cargo run --release -- run \
  --seed ./examples/seed.txt \
  --query "How will this policy affect public sentiment in 30 days?" \
  --agents 200

# Start REST API server
cargo run --release -- serve --addr 0.0.0.0:8080
```

---

## Pipeline

```
Seed File
   │
   ▼
[seed]  ── parse & normalise ──► SeedDocument
   │
   ▼
[graph] ── entity/relation extraction ──► KnowledgeGraph (petgraph)
   │                                         │
   ▼                                         ▼
[agent] ── persona gen + memory init ──► AgentPool (N agents)
   │
   ▼
[sim]   ── tick loop (rayon parallel) ──► SimulationState
   │           ▲
   │           └── God's-eye variable injection
   ▼
[report] ── ReportAgent synthesis ──► PredictionReport + InteractiveWorld
   │
   ▼
[api]   ── REST / SSE ──► Client (CLI or frontend)
```

---

## Project Structure

```
teri/
├── Cargo.toml
├── .env.example
├── README.md
├── ARCHITECTURE.md
└── src/
    ├── main.rs          # CLI entry point (clap)
    ├── lib.rs           # Module declarations
    ├── seed/            # Seed ingestion & normalisation
    ├── graph/           # Knowledge graph (petgraph + LLM extraction)
    ├── agent/           # Agent pool, personas, memory
    ├── sim/             # Simulation engine (tick loop, rayon)
    ├── report/          # Report generation & world interaction
    ├── memory/          # Persistent memory (RocksDB)
    └── api/             # HTTP server (axum) + SSE streaming
```

---

## Configuration

All configuration via `.env` or environment variables. See [`.env.example`](.env.example).  
Teri is LLM-provider agnostic — any OpenAI-compatible endpoint works.

---

## Status

🚧 **Pre-alpha — scaffold only.**  
Module interfaces are defined; implementation is in progress.

---

## Acknowledgements

MiroFish by [BaiFu / 666ghj](https://github.com/666ghj) is the original reference implementation.  
Simulation design draws on [OASIS](https://github.com/camel-ai/oasis) from the CAMEL-AI team.

---

## License

MIT
