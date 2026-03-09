# Teri Development TODO

> **Status:** Pre-alpha → Production-ready
> **Last Updated:** 2026-03-09

This checklist tracks end-to-end development of Teri, organized by implementation phase. Check off tasks as completed.

---

## Phase 0: Project Foundation

### Directory Structure
- [x] Create `src/` directory
- [x] Move `main.rs` and `lib.rs` into `src/`
- [x] Create module subdirectories:
  - [x] `src/seed/`
  - [x] `src/graph/`
  - [x] `src/agent/`
  - [x] `src/sim/`
  - [x] `src/memory/`
  - [x] `src/report/`
  - [x] `src/api/`
- [x] Create `templates/` directory for prompt templates
- [x] Create `examples/` directory with sample seed files
- [x] Create `data/` directory for persistent storage

### Configuration & Environment
- [x] Implement configuration loader using `config` crate
- [x] Create `Config` struct with all settings from `.env.example`
- [x] Add config validation on startup
- [x] Set up `tracing-subscriber` with env-filter
- [x] Create logging utilities module

### LLM Client Abstraction (Critical Path)
- [x] Define `LlmClient` trait in `src/llm.rs` (completely provider-agnostic)
  - [x] `async fn complete(&self, prompt: &str) -> Result<String>`
  - [x] `async fn complete_json<T>(&self, prompt: &str) -> Result<T>`
  - [x] `async fn stream(&self, prompt: &str) -> Result<impl Stream<Item = String>>`
- [x] Implement adapter pattern for provider-specific implementations
- [x] Implement `OpenAiAdapter` (for OpenAI chat completions API format)
  - [x] Constructor with base_url, api_key, model
  - [x] HTTP client using `reqwest`
  - [x] JSON mode support for structured outputs
  - [x] Streaming response handling (simplified, TODO: proper SSE)
  - [x] Error handling and retries with exponential backoff
- [x] Document adapter pattern and zero vendor lock-in design
- [x] Add LLM client tests with mock responses
- [x] Implement proper SSE streaming (OpenAI, Anthropic, Gemini adapters)
- [x] Add example adapters in documentation (Anthropic, llama.cpp)

### Error Handling
- [x] Define custom error types using `thiserror`
  - [x] `SeedError`
  - [x] `GraphError`
  - [x] `AgentError`
  - [x] `SimError`
  - [x] `MemoryError`
  - [x] `ReportError`
  - [x] `ApiError`
- [x] Create error conversion implementations
- [x] Add error context helpers

---

## Phase 1: Seed Module

### Core Types (`src/seed/mod.rs`)
- [x] Define `SeedDocument` struct
  - [x] `id: Uuid`
  - [x] `raw_text: String`
  - [x] `metadata: HashMap<String, String>`
  - [x] `created_at: DateTime<Utc>`
- [x] Define `SeedIngestor` struct
- [x] Implement `SeedIngestor::from_file(path: &str) -> Result<SeedDocument>`
- [x] Implement `SeedIngestor::from_url(url: &str) -> Result<SeedDocument>`

### File Format Support
- [ ] **Plain text** - Direct passthrough
  - [ ] Read file to string
  - [ ] Extract basic metadata (filename, size, modified date)
- [ ] **PDF** - Using `pdf-extract` or `lopdf`
  - [ ] Extract text content
  - [ ] Handle multi-page documents
  - [ ] Extract PDF metadata (author, title, etc.)
- [ ] **Web content** - Using `reqwest` + `scraper`
  - [ ] Fetch HTML content
  - [ ] Extract main text (remove nav, ads, etc.)
  - [ ] Extract metadata (title, description, author)
- [ ] **JSON** - Structured data
  - [ ] Parse and normalize to text
  - [ ] Preserve structure in metadata

### Testing
- [ ] Unit tests for each file format
- [ ] Test error handling (missing files, malformed PDFs, etc.)
- [ ] Integration test with sample files in `examples/`

---

## Phase 2: Graph Module

### Core Types (`src/graph/mod.rs`)
- [ ] Define `Entity` struct
  - [ ] `id: Uuid`
  - [ ] `name: String`
  - [ ] `kind: EntityKind` (enum: Person, Organization, Location, Concept, etc.)
- [ ] Define `Relation` struct
  - [ ] `kind: RelationKind` (enum: WorksFor, LocatedIn, RelatedTo, etc.)
  - [ ] `weight: f32`
- [ ] Define `KnowledgeGraph` struct
  - [ ] `inner: petgraph::Graph<Entity, Relation>`
  - [ ] `index: HashMap<String, NodeIndex>`

### Graph Construction
- [ ] Implement `KnowledgeGraph::new() -> Self`
- [ ] Implement `KnowledgeGraph::build(doc: &SeedDocument, llm: &dyn LlmClient) -> Result<Self>`
  - [ ] Design LLM prompt for entity extraction
  - [ ] Parse LLM JSON response into entities
  - [ ] Design LLM prompt for relation extraction
  - [ ] Parse LLM JSON response into relations
  - [ ] Build petgraph from entities and relations
  - [ ] Create name-to-index mapping
- [ ] Implement graph query methods
  - [ ] `get_entity(&self, name: &str) -> Option<&Entity>`
  - [ ] `get_neighbors(&self, entity_id: Uuid) -> Vec<&Entity>`
  - [ ] `get_subgraph(&self, entity_id: Uuid, depth: usize) -> KnowledgeGraph`

### Serialization
- [ ] Implement graph serialization to disk (optional feature)
- [ ] Implement graph deserialization from disk

### Testing
- [ ] Unit tests for graph construction
- [ ] Test entity/relation extraction with mock LLM
- [ ] Test graph query methods
- [ ] Integration test with real seed document

---

## Phase 3: Agent Module

### Core Types (`src/agent/mod.rs`)
- [ ] Define `Persona` struct
  - [ ] `name: String`
  - [ ] `background: String`
  - [ ] `traits: Vec<String>`
  - [ ] `role: String`
- [ ] Define `AgentState` enum (Idle, Thinking, Acting, etc.)
- [ ] Define `MemoryEntry` struct
  - [ ] `timestamp: DateTime<Utc>`
  - [ ] `content: String`
  - [ ] `importance: f32`
- [ ] Define `AgentMemory` struct
  - [ ] `short_term: VecDeque<MemoryEntry>` (capped at N entries)
  - [ ] `long_term_db: Arc<RocksDB>` (reference to shared DB)
- [ ] Define `Agent` struct
  - [ ] `id: Uuid`
  - [ ] `persona: Persona`
  - [ ] `memory: AgentMemory`
  - [ ] `state: AgentState`

### Persona Generation
- [ ] Create persona template in `templates/persona_gen.jinja`
- [ ] Implement `PersonaGenerator` struct
- [ ] Implement `PersonaGenerator::generate(graph: &KnowledgeGraph, llm: &dyn LlmClient) -> Result<Persona>`
  - [ ] Sample entity from graph as role anchor
  - [ ] Generate persona using LLM with template
  - [ ] Parse and validate persona

### Agent Pool
- [ ] Define `AgentPool` struct
  - [ ] `agents: Vec<Agent>`
  - [ ] `group_memory: Arc<RwLock<Vec<MemoryEntry>>>`
- [ ] Implement `AgentPool::spawn(n: usize, graph: &KnowledgeGraph, llm: &dyn LlmClient) -> Result<Self>`
  - [ ] Generate N unique personas
  - [ ] Initialize agent memory
  - [ ] Create shared group memory
- [ ] Implement `AgentPool::get(&self, id: Uuid) -> Option<&Agent>`
- [ ] Implement `AgentPool::get_mut(&mut self, id: Uuid) -> Option<&mut Agent>`

### Agent Actions
- [ ] Define `Action` enum (Speak, Move, Interact, Observe, etc.)
- [ ] Create action template in `templates/agent_action.jinja`
- [ ] Implement `Agent::step(&mut self, world: &WorldState, llm: &dyn LlmClient) -> Result<Action>`
  - [ ] Retrieve relevant memories
  - [ ] Construct context from world state + memories
  - [ ] Generate action using LLM
  - [ ] Parse and validate action
  - [ ] Store action in memory

### Testing
- [ ] Unit tests for persona generation
- [ ] Unit tests for agent memory operations
- [ ] Unit tests for agent action generation
- [ ] Test agent pool spawning
- [ ] Integration test with mock world state

---

## Phase 4: Simulation Module

### Core Types (`src/sim/mod.rs`)
- [ ] Define `WorldState` struct
  - [ ] `tick: u32`
  - [ ] `agents: HashMap<Uuid, AgentSnapshot>` (lightweight agent state)
  - [ ] `events: Vec<Event>` (actions taken this tick)
  - [ ] `variables: HashMap<String, f32>` (God's-eye variables)
- [ ] Define `Event` struct
  - [ ] `agent_id: Uuid`
  - [ ] `action: Action`
  - [ ] `timestamp: DateTime<Utc>`
- [ ] Define `WorldSnapshot` struct (serializable tick state)
- [ ] Define `SimConfig` struct
  - [ ] `max_ticks: u32`
  - [ ] `parallelism: usize`
  - [ ] `inject_fn: Option<Box<dyn Fn(u32, &mut WorldState) + Send + Sync>>`
- [ ] Define `SimulationResult` struct
  - [ ] `id: Uuid`
  - [ ] `history: Vec<WorldSnapshot>`
  - [ ] `final_state: WorldState`

### Simulation Engine
- [ ] Implement `SimEngine` struct
- [ ] Implement `SimEngine::new(config: SimConfig) -> Self`
- [ ] Implement `SimEngine::run(pool: &mut AgentPool, graph: &KnowledgeGraph, llm: &dyn LlmClient) -> Result<SimulationResult>`
  - [ ] Initialize world state
  - [ ] Create broadcast channel for streaming
  - [ ] Tick loop:
    - [ ] Parallel agent step using `rayon::par_iter`
    - [ ] Collect actions from all agents
    - [ ] Apply actions to world state
    - [ ] Call inject_fn if present
    - [ ] Create snapshot
    - [ ] Broadcast snapshot to listeners
    - [ ] Store snapshot in history
  - [ ] Return simulation result

### State Management
- [ ] Implement `WorldState::new() -> Self`
- [ ] Implement `WorldState::apply(&mut self, actions: Vec<Action>)`
- [ ] Implement `WorldState::snapshot(&self) -> WorldSnapshot`
- [ ] Implement `WorldState::inject_variable(&mut self, key: String, value: f32)`

### Streaming Support
- [ ] Set up `tokio::sync::broadcast` channel
- [ ] Implement snapshot streaming to API layer
- [ ] Handle backpressure for slow consumers

### Testing
- [ ] Unit tests for world state operations
- [ ] Unit tests for action application
- [ ] Test parallel agent execution
- [ ] Test God's-eye injection
- [ ] Integration test with small agent pool

---

## Phase 5: Memory Module

### Core Types (`src/memory/mod.rs`)
- [ ] Define `MemoryStore` struct
  - [ ] `db: Arc<rocksdb::DB>`
- [ ] Define key schemas as constants
  - [ ] `agent:{uuid}:ltm:{timestamp}` → MemoryEntry
  - [ ] `world:{sim_id}:tick:{n}` → WorldSnapshot

### Store Implementation
- [ ] Implement `MemoryStore::new(path: &str) -> Result<Self>`
  - [ ] Open/create RocksDB instance
  - [ ] Configure column families if needed
- [ ] Implement agent memory methods
  - [ ] `write_ltm(&self, agent_id: Uuid, entry: &MemoryEntry) -> Result<()>`
  - [ ] `read_ltm(&self, agent_id: Uuid, limit: usize) -> Result<Vec<MemoryEntry>>`
  - [ ] `query_ltm(&self, agent_id: Uuid, query: &str) -> Result<Vec<MemoryEntry>>` (stub for now)
- [ ] Implement world snapshot methods
  - [ ] `write_snapshot(&self, sim_id: Uuid, tick: u32, snapshot: &WorldSnapshot) -> Result<()>`
  - [ ] `read_snapshot(&self, sim_id: Uuid, tick: u32) -> Result<WorldSnapshot>`
  - [ ] `read_history(&self, sim_id: Uuid) -> Result<Vec<WorldSnapshot>>`

### Async Integration
- [ ] Wrap blocking RocksDB calls in `tokio::task::spawn_blocking`
- [ ] Create async wrapper methods for all store operations

### Testing
- [ ] Unit tests for memory write/read
- [ ] Test snapshot persistence
- [ ] Test concurrent access
- [ ] Test error handling (corrupted DB, etc.)

### Future: Vector Search (Stub)
- [ ] Add placeholder for vector similarity search
- [ ] Document integration points for `hnswlib-rs` or FAISS

---

## Phase 6: Report Module

### Core Types (`src/report/mod.rs`)
- [ ] Define `TimelineEvent` struct
  - [ ] `tick: u32`
  - [ ] `description: String`
  - [ ] `significance: f32`
- [ ] Define `AgentHighlight` struct
  - [ ] `agent_id: Uuid`
  - [ ] `agent_name: String`
  - [ ] `summary: String`
- [ ] Define `PredictionReport` struct
  - [ ] `summary: String`
  - [ ] `timeline: Vec<TimelineEvent>`
  - [ ] `agent_highlights: Vec<AgentHighlight>`
  - [ ] `confidence: f32`
  - [ ] `raw_query: String`

### Report Agent
- [ ] Define `ReportAgent` struct
- [ ] Create report template in `templates/report_gen.jinja`
- [ ] Implement `ReportAgent::generate(result: &SimulationResult, query: &str, llm: &dyn LlmClient) -> Result<PredictionReport>`
  - [ ] Analyze simulation history
  - [ ] Extract key timeline events
  - [ ] Identify significant agents
  - [ ] Generate summary using LLM
  - [ ] Calculate confidence score
  - [ ] Assemble report struct

### Streaming Report Generation
- [ ] Implement `ReportAgent::generate_stream()` for SSE
  - [ ] Stream LLM response chunks
  - [ ] Yield partial report updates

### Interactive Chat
- [ ] Define `ChatMessage` struct
- [ ] Create chat template in `templates/agent_chat.jinja`
- [ ] Implement `ReportAgent::chat(message: &str, context: &SimulationResult, llm: &dyn LlmClient) -> Result<String>`
  - [ ] Parse user intent (query world, query agent, etc.)
  - [ ] Retrieve relevant context
  - [ ] Generate response using LLM
- [ ] Implement agent-specific chat
  - [ ] `Agent::chat(message: &str, llm: &dyn LlmClient) -> Result<String>`
  - [ ] Include persona + memory in context

### Testing
- [ ] Unit tests for report generation
- [ ] Test timeline extraction
- [ ] Test agent highlight selection
- [ ] Test chat with mock simulation result

---

## Phase 7: API Module

### Core Types (`src/api/mod.rs`)
- [ ] Define request/response structs
  - [ ] `CreateSimRequest` (seed, query, agent_count)
  - [ ] `CreateSimResponse` (sim_id, status)
  - [ ] `SimStatusResponse` (sim_id, tick, status, agent_count)
  - [ ] `InjectRequest` (variable, value)
  - [ ] `ChatRequest` (message, agent_id optional)
  - [ ] `ChatResponse` (message)

### API Server
- [ ] Define `ApiState` struct
  - [ ] `sims: Arc<RwLock<HashMap<Uuid, SimulationHandle>>>`
  - [ ] `llm_client: Arc<dyn LlmClient>`
  - [ ] `memory_store: Arc<MemoryStore>`
  - [ ] `config: Config`
- [ ] Define `SimulationHandle` struct
  - [ ] `id: Uuid`
  - [ ] `status: SimStatus` (Running, Completed, Failed)
  - [ ] `result: Option<SimulationResult>`
  - [ ] `broadcast_rx: broadcast::Receiver<WorldSnapshot>`

### Endpoints
- [ ] `POST /sim` - Create and start simulation
  - [ ] Parse request
  - [ ] Ingest seed
  - [ ] Build graph
  - [ ] Spawn agents
  - [ ] Start simulation in background task
  - [ ] Return sim_id
- [ ] `GET /sim/:id` - Get simulation status
  - [ ] Look up simulation
  - [ ] Return current status
- [ ] `GET /sim/:id/stream` - SSE stream of live ticks
  - [ ] Subscribe to broadcast channel
  - [ ] Stream snapshots as SSE events
  - [ ] Handle client disconnect
- [ ] `GET /sim/:id/report` - Retrieve final report
  - [ ] Check simulation completed
  - [ ] Generate report if not cached
  - [ ] Return report JSON
- [ ] `POST /sim/:id/inject` - God's-eye variable injection
  - [ ] Validate simulation is running
  - [ ] Inject variable into world state
  - [ ] Return success
- [ ] `POST /sim/:id/chat` - Chat with agent or ReportAgent
  - [ ] Parse message and target
  - [ ] Route to appropriate chat handler
  - [ ] Return response

### Middleware
- [ ] Set up `tower-http` CORS
- [ ] Set up `tower-http` tracing middleware
- [ ] Add request ID generation
- [ ] Add error handling middleware

### Server Lifecycle
- [ ] Implement `serve(config: Config) -> Result<()>`
  - [ ] Initialize API state
  - [ ] Build axum router
  - [ ] Bind to configured address
  - [ ] Graceful shutdown on SIGTERM

### Testing
- [ ] Integration tests for each endpoint
- [ ] Test SSE streaming
- [ ] Test concurrent simulations
- [ ] Test error responses

---

## Phase 8: CLI Integration

### Main Entry Point (`src/main.rs`)
- [ ] Implement `Commands::Run` handler
  - [ ] Load config
  - [ ] Initialize LLM client
  - [ ] Initialize memory store
  - [ ] Run full pipeline:
    - [ ] Ingest seed
    - [ ] Build graph
    - [ ] Spawn agents
    - [ ] Run simulation
    - [ ] Generate report
  - [ ] Print report to stdout
  - [ ] Save report to file
- [ ] Implement `Commands::Serve` handler
  - [ ] Load config
  - [ ] Start API server
  - [ ] Handle graceful shutdown

### Additional CLI Commands
- [ ] Add `inspect` subcommand
  - [ ] Inspect saved simulation
  - [ ] Query specific tick
  - [ ] Query specific agent
- [ ] Add `replay` subcommand
  - [ ] Replay simulation from snapshots
  - [ ] Interactive tick-by-tick navigation

### Testing
- [ ] End-to-end test with example seed file
- [ ] Test CLI argument parsing
- [ ] Test error handling

---

## Phase 9: Templates & Prompts

### Persona Templates (`templates/`)
- [ ] `persona_gen.jinja` - Agent persona generation
  - [ ] Include graph context
  - [ ] Specify output format (JSON)
- [ ] `agent_action.jinja` - Agent action generation
  - [ ] Include persona, memory, world state
  - [ ] Specify action format
- [ ] `report_gen.jinja` - Report synthesis
  - [ ] Include simulation history
  - [ ] Specify report structure
- [ ] `agent_chat.jinja` - Agent chat responses
  - [ ] Include persona and memory
  - [ ] Natural conversation style

### Template Loading
- [ ] Implement template loader using `minijinja`
- [ ] Cache compiled templates
- [ ] Add template validation on startup

### Testing
- [ ] Test template rendering with sample data
- [ ] Validate JSON output from templates

---

## Phase 10: Examples & Documentation

### Example Files (`examples/`)
- [ ] `seed.txt` - Simple text seed
- [ ] `policy.pdf` - Sample policy document (if available)
- [ ] `news.json` - Structured news data
- [ ] `web_article.html` - Sample HTML content

### Documentation
- [ ] Update README with:
  - [ ] Installation instructions
  - [ ] Detailed usage examples
  - [ ] Configuration guide
  - [ ] API documentation link
- [ ] Create `DEVELOPMENT.md`
  - [ ] Development setup
  - [ ] Running tests
  - [ ] Contributing guidelines
- [ ] Create `API.md`
  - [ ] Endpoint documentation
  - [ ] Request/response examples
  - [ ] SSE event format
- [ ] Add inline code documentation
  - [ ] Module-level docs
  - [ ] Public API docs
  - [ ] Examples in doc comments

---

## Phase 11: Testing & Quality

### Unit Tests
- [ ] Achieve >80% code coverage
- [ ] Test all error paths
- [ ] Test edge cases (empty graphs, single agent, etc.)

### Integration Tests
- [ ] End-to-end pipeline test
- [ ] Multi-simulation concurrency test
- [ ] Memory persistence test
- [ ] API integration test

### Performance Tests
- [ ] Benchmark agent step execution
- [ ] Benchmark graph construction
- [ ] Benchmark parallel simulation
- [ ] Profile memory usage

### Code Quality
- [ ] Run `clippy` and fix all warnings
- [ ] Run `rustfmt` on all code
- [ ] Add CI/CD pipeline (GitHub Actions)
  - [ ] Build on push
  - [ ] Run tests
  - [ ] Run clippy
  - [ ] Check formatting

---

## Phase 12: Production Readiness

### Performance Optimization
- [ ] Profile with `cargo flamegraph`
- [ ] Optimize hot paths identified in profiling
- [ ] Tune rayon thread pool size
- [ ] Optimize LLM prompt sizes
- [ ] Add caching where appropriate

### Error Handling & Resilience
- [ ] Add retry logic for LLM calls
- [ ] Add timeout handling
- [ ] Add graceful degradation for non-critical failures
- [ ] Improve error messages for users

### Monitoring & Observability
- [ ] Add structured logging with context
- [ ] Add metrics collection (optional)
- [ ] Add health check endpoint
- [ ] Add simulation progress tracking

### Deployment
- [ ] Create Dockerfile
- [ ] Create docker-compose.yml for local dev
- [ ] Document deployment options
  - [ ] Single binary deployment
  - [ ] Docker deployment
  - [ ] Cloud deployment (AWS, GCP, etc.)
- [ ] Add deployment scripts

### Security
- [ ] Validate all user inputs
- [ ] Sanitize LLM outputs
- [ ] Add rate limiting to API
- [ ] Secure API key handling
- [ ] Add HTTPS support

---

## Phase 13: Future Enhancements (Post-MVP)

### Local LLM Support
- [ ] Research `candle` integration
- [ ] Research `llama.cpp` FFI
- [ ] Implement local LLM client
- [ ] Add model loading/management

### Vector Memory
- [ ] Integrate `hnswlib-rs` or FAISS
- [ ] Implement embedding generation
- [ ] Implement semantic memory retrieval
- [ ] Update agent context retrieval to use vectors

### WASM Target
- [ ] Evaluate WASM compatibility
- [ ] Create WASM build target
- [ ] Create browser demo
- [ ] Document WASM limitations

### Distributed Simulation
- [ ] Design agent sharding strategy
- [ ] Implement RPC layer (`tarpc` or gRPC)
- [ ] Implement distributed world state
- [ ] Test multi-node simulation

### Frontend
- [ ] Create separate frontend repo
- [ ] Implement real-time visualization
- [ ] Implement interactive controls
- [ ] Implement agent chat interface

---

## Completion Criteria

**MVP Ready:**
- ✅ All Phase 0-8 tasks completed
- ✅ End-to-end pipeline working
- ✅ API server functional
- ✅ Basic tests passing
- ✅ Example simulations run successfully

**Production Ready:**
- ✅ All Phase 0-12 tasks completed
- ✅ >80% test coverage
- ✅ Performance benchmarks met
- ✅ Documentation complete
- ✅ Deployment guide ready

**Future-Proof:**
- ✅ Phase 13 roadmap documented
- ✅ Extension points identified
- ✅ Community contribution guidelines ready
