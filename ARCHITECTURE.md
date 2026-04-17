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

**Teri is completely provider-agnostic.** All LLM calls go through a single `LlmClient` trait that makes **zero assumptions** about the underlying provider:

```rust
#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn complete(&self, prompt: &str) -> Result<String>;
    async fn complete_json<T: DeserializeOwned>(&self, prompt: &str) -> Result<T>;
    async fn stream(&self, prompt: &str) -> Result<impl Stream<Item = String>>;
}
```

**Adapter Pattern:** Teri uses provider-specific adapters that implement `LlmClient`. Each adapter handles the API-specific details:

**Included Adapters:**
- `OpenAiAdapter` - For OpenAI chat completions API format
  - Works with: OpenAI, Ollama, LM Studio, vLLM, Together AI, Groq, etc.
- `AnthropicAdapter` - For Anthropic Claude (Messages API)
  - Works with: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku, etc.
- `GeminiAdapter` - For Google Gemini (generateContent API)
  - Works with: Gemini 1.5 Pro, Gemini 1.5 Flash, etc.

**Usage Example:**

```rust
use teri::{LlmClient, OpenAiAdapter, AnthropicAdapter, GeminiAdapter};

// Use OpenAI (or compatible)
let openai = OpenAiAdapter::new(&config);
let response = openai.complete("Hello!").await?;

// Use Anthropic Claude
let claude = AnthropicAdapter::new("sk-ant-...".to_string(), "claude-3-5-sonnet-20241022".to_string());
let response = claude.complete("Hello!").await?;

// Use Google Gemini
let gemini = GeminiAdapter::new("AIza...".to_string(), "gemini-1.5-pro".to_string());
let response = gemini.complete("Hello!").await?;
```

**Add Your Own Adapter:**

```rust
// Example: Local llama.cpp adapter (no HTTP, no external API)
pub struct LlamaCppAdapter {
    model_path: PathBuf,
}

#[async_trait]
impl LlmClient for LlamaCppAdapter {
    async fn complete(&self, prompt: &str) -> Result<String> {
        // Call llama.cpp directly via FFI
        // No network calls, fully local
    }
}
```

**No vendor lock-in.** The core simulation engine only depends on the `LlmClient` trait, never on specific adapters.

---

## Community Seed Abstraction

**Teri accepts live community knowledge as a seed source**, parallel to document ingestion. All community platform access goes through a single `CommunityAdapter` trait — no assumptions about the underlying platform.

```rust
#[async_trait]
pub trait CommunityAdapter: Send + Sync {
    async fn fetch_domains(&self)                          -> Result<Vec<CommunityDomain>>;
    async fn fetch_contributors(&self, domain: &str)      -> Result<Vec<CommunityContributor>>;
    async fn fetch_signal(&self, domain: &str)            -> Result<CommunitySignal>;
    async fn fetch_topics(&self, domain: &str)            -> Result<Vec<CommunityTopic>>;
    async fn to_seed_document(&self)                      -> Result<SeedDocument>;
}

// Normalized output types — platform-agnostic
pub struct CommunityDomain {
    pub slug: String,
    pub name: String,
}

pub struct CommunityContributor {
    pub handle: String,
    pub domain_weight: f32,   // contribution weight, karma, reputation — normalized to [0,1]
    pub verified: bool,
    pub tenure_days: u32,
}

pub struct CommunitySignal {
    pub domain: String,
    pub message_volume: u32,
    pub contributor_count: u32,
    pub topic_velocity: f32,  // open/resolve rate or equivalent
    pub sentiment_proxy: f32, // net reaction ratio or equivalent
}

pub struct CommunityTopic {
    pub id: String,
    pub name: String,
    pub status: String,
    pub message_count: u32,
    pub last_active: DateTime<Utc>,
}
```

**Adapter Pattern:** Each platform adapter implements `CommunityAdapter` and handles its own API mapping. The simulation engine only depends on the trait.

**Signal quality varies by platform.** Each adapter normalizes to the common types:

| Platform | Domain signal | Contributor weight | Topic structure |
|---|---|---|---|
| Pebesen | Native, structured | Explicit weight | Stream/Topic hierarchy |
| Reddit | Subreddit mapping | Karma (noisy) | Flat threads |
| Zulip | Stream mapping | None native | Stream/Topic (close match) |
| Discourse | Category mapping | Trust level | Thread |
| Stack Overflow | Tag mapping | Reputation | Q&A |

**Module structure:**

```
src/seed/
├── mod.rs           # SeedIngestor (document-based)
└── community/
    ├── mod.rs       # CommunityAdapter trait + normalized types
    ├── pebesen.rs   # PebesenAdapter (reference implementation — highest signal fidelity)
    ├── reddit.rs    # RedditAdapter
    ├── zulip.rs     # ZulipAdapter
    └── discourse.rs # DiscourseAdapter
```

`PebesenAdapter` is the reference implementation because Pebesen's structured topic/domain/contributor model requires the least normalization logic, making it the cleanest example for new adapter authors.

**Add your own adapter:**

```rust
pub struct MyPlatformAdapter {
    base_url: String,
    api_key: String,
}

#[async_trait]
impl CommunityAdapter for MyPlatformAdapter {
    async fn fetch_signal(&self, domain: &str) -> Result<CommunitySignal> {
        // map your platform's native signals to CommunitySignal
    }
    // ...
}
```

**No platform lock-in.** The simulation engine accepts `&dyn CommunityAdapter` — swap platforms without changing simulation code.

---

## Community Feedback Abstraction

**Teri can write simulation output back to the source platform**, closing the loop from prediction to action. All write-back goes through a single `CommunityFeedback` trait — symmetric counterpart to `CommunityAdapter`.

```rust
#[async_trait]
pub trait CommunityFeedback: Send + Sync {
    async fn push_topic_signals(
        &self,
        signals: Vec<TopicSignal>,
    ) -> Result<()>;

    async fn push_contributor_trajectories(
        &self,
        trajectories: Vec<ContributorTrajectory>,
    ) -> Result<()>;

    async fn push_health_risks(
        &self,
        risks: Vec<SpaceHealthRisk>,
    ) -> Result<()>;
}

pub struct TopicSignal {
    pub domain: String,
    pub topic_name: String,
    pub confidence: f32,
    pub horizon_days: u32,
}

pub struct ContributorTrajectory {
    pub handle: String,
    pub domain: String,
    pub trajectory: String,   // "rising_anchor", "at_risk_churn", etc.
    pub confidence: f32,
}

pub struct SpaceHealthRisk {
    pub risk_type: String,    // "contributor_concentration", "topic_stagnation", etc.
    pub description: String,
    pub confidence: f32,
    pub horizon_days: u32,
}
```

**Calibration loop.** When a platform moderator marks a prediction as actioned, that event is the ground-truth signal. `PebesenFeedback` can optionally poll `POST /v1/intelligence/spaces/:slug/predictions/:id/action` to retrieve actioned timestamps and feed them back as confidence adjustments for the next simulation run against that community. Predictions confirmed accurate increase future confidence weighting; predictions that expired unactioned decrease it. The model self-calibrates per community over time.

**Adapter Pattern:** `CommunityFeedback` implementations are independent of `CommunityAdapter` implementations. A simulation can read from one platform and write to another, or use the same platform for both.

```
src/seed/community/
├── mod.rs           # CommunityAdapter + CommunityFeedback traits + all types
├── pebesen.rs       # PebesenAdapter + PebesenFeedback (reference implementations)
├── reddit.rs        # RedditAdapter (read-only — no feedback endpoint)
├── zulip.rs         # ZulipAdapter
└── discourse.rs     # DiscourseAdapter
```

**No coupling.** `CommunityFeedback` is additive — Teri functions without it. Write-back only activates when both systems have live deployments and a platform implements the receiving endpoint.

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
