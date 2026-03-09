# Teri

> **Rust-native Swarm Intelligence Prediction Engine**
> A ground-up rewrite of [MiroFish](https://github.com/666ghj/MiroFish) — designed for performance, type safety, and deployment simplicity.

**Name:** “Teri” (Indonesian: *ikan teri*) is the anchovy — one of the smallest fish in the sea, yet one of the most consequential. Anchovies move in vast, tightly coordinated schools: thousands of individuals following simple local rules, producing emergent behavior no single fish planned or directed. That is exactly what this engine does. Seed the world. Spawn the swarm. Watch emergence happen. It’s also a nod to Indonesian waters, where *ikan teri* has fed communities and ecosystems for centuries, punching far above its size.

---

## What is Teri?

Teri turns seed materials (news articles, policy drafts, financial signals, novels) into a **high-fidelity parallel digital world** populated by thousands of independent agents. Each agent carries its own persona, long-term memory, and behavioural logic. The swarm self-organises, and you observe — or intervene — from a God's-eye view.

**Input** → seed file + natural-language prediction query
**Output** → structured prediction report + interactive living simulation world

### Key Features

- 🧠 **Multi-Provider LLM Support** - OpenAI, Anthropic, Google Gemini, local models (Ollama, LM Studio)
- 🚀 **True Parallelism** - Rayon-powered agent simulation, no GIL limitations
- 💾 **Persistent Memory** - Rust-native redb for fast agent long-term memory
- 🌐 **Real-time Streaming** - SSE-based live simulation state updates
- 🎯 **Zero Vendor Lock-in** - Adapter pattern for any LLM provider
- 📦 **Single Binary** - No Docker, no venv, just `cargo build --release`

---

## Why Rust?

| Concern | Python (MiroFish) | Teri (Rust) |
| --- | --- | --- |
| Agent parallelism | GIL-limited threads | `rayon` true parallelism |
| Memory per agent | ~MB overhead | Controlled, stack-friendly |
| Deployment | Docker + venv | Single static binary |
| Type safety | Runtime errors | Compile-time guarantees |
| Async LLM calls | `asyncio` | `tokio` native |

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
    ├── memory/          # Persistent memory (redb)
    └── api/             # HTTP server (axum) + SSE streaming
```

---

## Configuration

All configuration via `.env` or environment variables. See [`.env.example`](.env.example).

### LLM Provider Support

**Teri is completely LLM-provider agnostic.** Choose any provider via adapter pattern:

- **OpenAI** (GPT-4, GPT-4o) - `OpenAiAdapter`
- **Anthropic** (Claude 3.5 Sonnet, Opus, Haiku) - `AnthropicAdapter`
- **Google** (Gemini 1.5 Pro, Flash) - `GeminiAdapter`
- **Local models** (Ollama, LM Studio, vLLM) - `OpenAiAdapter` (OpenAI-compatible)
- **Custom providers** - Implement the `LlmClient` trait

No vendor lock-in. Swap providers without changing simulation code.

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
