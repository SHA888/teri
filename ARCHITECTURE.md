# Teri — Architecture

## Overview

Teri is a multi-stage pipeline. Each stage is an independent Rust module with a clean async interface. Stages communicate via typed structs — no stringly-typed JSON blobs until the API boundary.

```
┌────────────┐    ┌────────────┐    ┌────────────┐
│    seed    │───►│   graph    │───►│   agent    │
│  ingestor  │    │  builder   │    │    pool    │
└────────────┘    └────────────┘    └─────┬──────┘
                                          │
                                          ▼
                                   ┌────────────┐
                                   │    sim     │◄── God's-eye injection
                                   │   engine   │
                                   └─────┬──────┘
                                         │
                              ┌──────────┴──────────┐
                              ▼                     ▼
                       ┌────────────┐        ┌────────────┐
                       │   report   │        │  memory    │
                       │   agent    │        │   store    │
                       └─────┬──────┘        └────────────┘
                             │
                             ▼
                       ┌────────────┐
                       │    api     │
                       │ (axum/SSE) │
                       └────────────┘
```

---

## Module Contracts

### `seed` — Ingestor

**Responsibility:** Accept raw input (path or URL), normalise to `SeedDocument`.

```rust
pub struct SeedDocument {
    pub id: Uuid,
    pub raw_text: String,
    pub metadata: HashMap<String, String>,
    pub created_at: DateTime<Utc>,
}
```

**Key decisions:**
- PDF extraction via `pdf-extract` or `lopdf`
- Web content via `reqwest` + `scraper`
- Plain text / JSON passed through directly

---

### `graph` — Knowledge Graph

**Responsibility:** Extract entities and relations from `SeedDocument`, build a `KnowledgeGraph`.

```rust
pub struct KnowledgeGraph {
    inner: petgraph::Graph<Entity, Relation>,
    index: HashMap<String, NodeIndex>,
}

pub struct Entity { pub id: Uuid, pub name: String, pub kind: EntityKind }
pub struct Relation { pub kind: RelationKind, pub weight: f32 }
```

**Key decisions:**
- Entity/relation extraction via LLM (structured output, JSON mode)
- Graph stored in-memory during sim; optionally serialised to disk
- GraphRAG: entity neighbourhood used to ground agent context windows

---

### `agent` — Agent Pool

**Responsibility:** Spawn N agents, each with a generated persona and initialised memory.

```rust
pub struct Agent {
    pub id: Uuid,
    pub persona: Persona,
    pub memory: AgentMemory,
    pub state: AgentState,
}

pub struct Persona {
    pub name: String,
    pub background: String,
    pub traits: Vec<String>,
    pub role: String,              // derived from graph entities
}
```

**Key decisions:**
- Persona templates rendered via `minijinja` — keeps prompt logic out of Rust code
- Each agent holds a short-term (in-memory VecDeque) and long-term (RocksDB) memory
- Group memory is a shared RwLock<Vec<MemoryEntry>> per cluster

---

### `sim` — Simulation Engine

**Responsibility:** Run the tick-based interaction loop. Manage time-series state. Support God's-eye variable injection mid-run.

```rust
pub struct SimConfig {
    pub max_ticks: u32,
    pub parallelism: usize,
    pub inject_fn: Option<Box<dyn Fn(u32, &mut WorldState) + Send + Sync>>,
}
```

**Key decisions:**
- `rayon::par_iter` for parallel agent step execution within each tick
- Each tick: agents observe world state → LLM generates action → world state updated
- `tokio::sync::broadcast` channel for real-time state streaming to API layer
- Tick state snapshots stored to enable replay

**Tick loop pseudocode:**
```
for tick in 0..config.max_ticks {
    let actions = agent_pool.par_step(&world_state, tick);
    world_state.apply(actions);
    if let Some(f) = inject_fn { f(tick, &mut world_state); }
    tx.send(world_state.snapshot());
}
```

---

### `memory` — Persistent Store

**Responsibility:** RocksDB-backed key-value store for agent long-term memory and world tick snapshots.

**Key schema:**
```
agent:{uuid}:ltm:{timestamp}  → MemoryEntry (JSON)
world:{sim_id}:tick:{n}       → WorldSnapshot (bincode)
```

**Key decisions:**
- Short-term memory stays in-process (`VecDeque<MemoryEntry>`, capped)
- Long-term memory written async, read on context retrieval
- Vector similarity search deferred — stub for now, pluggable (hnswlib-rs / faiss FFI)

---

### `report` — Report Agent

**Responsibility:** Post-simulation synthesis. Produce `PredictionReport`. Enable interactive world chat.

```rust
pub struct PredictionReport {
    pub summary: String,
    pub timeline: Vec<TimelineEvent>,
    pub agent_highlights: Vec<AgentHighlight>,
    pub confidence: f32,
    pub raw_query: String,
}
```

**Key decisions:**
- ReportAgent has tool access: query world state, query specific agents, access tick history
- Streamed generation via SSE (chunked LLM response forwarded to client)
- Interactive chat: user message → agent lookup → LLM with persona + memory context

---

### `api` — HTTP Server

**Responsibility:** REST API + SSE for frontend or CLI consumers.

**Planned endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/sim` | Create and start a simulation |
| `GET`  | `/sim/:id` | Get simulation status |
| `GET`  | `/sim/:id/stream` | SSE stream of live tick state |
| `GET`  | `/sim/:id/report` | Retrieve final report |
| `POST` | `/sim/:id/inject` | God's-eye variable injection |
| `POST` | `/sim/:id/chat` | Chat with an agent or ReportAgent |

**Key decisions:**
- `axum` router with `tower-http` CORS + tracing middleware
- SSE via `axum::response::Sse` + `tokio::sync::broadcast`
- All request/response types are `serde` structs — no raw JSON manipulation

---

## Data Flow (typed)

```
&str (path)
  │  seed::SeedIngestor::from_file()
  ▼
SeedDocument
  │  graph::KnowledgeGraph::build(&doc)
  ▼
KnowledgeGraph
  │  agent::AgentPool::spawn(n, &graph)
  ▼
AgentPool
  │  sim::SimEngine::run(&pool, &graph, config)
  ▼
SimulationResult { history: Vec<WorldSnapshot>, final_state: WorldState }
  │  report::ReportAgent::generate(&result, &query)
  ▼
PredictionReport
```

---

## LLM Abstraction

All LLM calls go through a single `LlmClient` trait to stay provider-agnostic:

```rust
#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn complete(&self, prompt: &str) -> Result<String>;
    async fn complete_json<T: DeserializeOwned>(&self, prompt: &str) -> Result<T>;
    async fn stream(&self, prompt: &str) -> Result<impl Stream<Item = String>>;
}
```

Concrete implementation: `OpenAiClient` (reqwest). Swap for local (Ollama, etc.) anytime.

---

## Concurrency Model

```
tokio runtime (async I/O — LLM calls, API, disk)
      │
      └── rayon threadpool (CPU — parallel agent steps per tick)
```

These are kept strictly separate: async tasks yield on I/O, rayon handles CPU-bound parallelism. No blocking calls inside tokio tasks.

---

## Crate Choices

| Crate | Role | Why |
|---|---|---|
| `tokio` | Async runtime | Ecosystem standard |
| `axum` | HTTP | Ergonomic, tower-native |
| `rayon` | CPU parallelism | Zero-cost agent par_iter |
| `petgraph` | Knowledge graph | Proven, flexible |
| `rocksdb` | Persistent memory | Embedded, fast KV |
| `minijinja` | Prompt templates | Keeps prompts in files, not Rust strings |
| `clap` | CLI | Derive macros, clean UX |
| `anyhow` / `thiserror` | Error handling | anyhow for bins, thiserror for lib types |

---

## Future Considerations

- **Local LLM support** — `candle` (Rust-native inference) or `llama.cpp` FFI
- **Vector memory** — `hnswlib-rs` for semantic retrieval from long-term memory
- **WASM target** — `sim` core compiled to WASM for browser sandbox
- **Distributed sim** — agent sharding across nodes via `tarpc` or gRPC
- **Frontend** — Separate repo; communicates via API + SSE (any JS framework)
