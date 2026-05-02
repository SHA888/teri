use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::broadcast;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Action {
    Speak(String),
    Move(String),
    Interact(String),
    Observe(String),
    Think(String),
}

impl std::fmt::Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Action::Speak(content) => write!(f, "Spoke: {}", content),
            Action::Move(location) => write!(f, "Moved to: {}", location),
            Action::Interact(target) => write!(f, "Interacted with: {}", target),
            Action::Observe(target) => write!(f, "Observed: {}", target),
            Action::Think(content) => write!(f, "Thought: {}", content),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub agent_id: Uuid,
    pub action: Action,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSnapshot {
    pub id: Uuid,
    pub name: String,
    pub state: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldState {
    pub tick: u32,
    pub agents: HashMap<Uuid, AgentSnapshot>,
    pub events: Vec<Event>,
    pub variables: HashMap<String, f32>,
}

impl WorldState {
    pub fn new() -> Self {
        Self {
            tick: 0,
            // Pre-allocate with typical small-pool capacity to avoid early rehashing.
            agents: HashMap::with_capacity(16),
            events: Vec::with_capacity(16),
            variables: HashMap::with_capacity(8),
        }
    }

    pub fn add_agent_snapshot(&mut self, agent_id: Uuid, snapshot: AgentSnapshot) {
        self.agents.insert(agent_id, snapshot);
    }

    pub fn add_event(&mut self, event: Event) {
        self.events.push(event);
    }

    pub fn inject_variable(&mut self, key: String, value: f32) {
        self.variables.insert(key, value);
    }

    pub fn get_variable(&self, key: &str) -> Option<f32> {
        self.variables.get(key).copied()
    }

    pub fn apply(&mut self, agent_id: Uuid, action: Action) {
        self.apply_at(agent_id, action, chrono::Utc::now());
    }

    pub fn apply_at(
        &mut self,
        agent_id: Uuid,
        action: Action,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) {
        let event = Event { agent_id, action, timestamp };
        self.events.push(event);
    }

    /// Advance to the next tick, clearing per-tick events.
    ///
    /// Invariant: `events` must contain at most one entry per registered agent.
    /// Callers (SimEngine) are responsible for enforcing this; violations are
    /// caught in debug builds via the assert below.
    pub fn advance_tick(&mut self) {
        debug_assert!(
            self.events.len() <= self.agents.len().max(1) * 2,
            "events ({}) exceeded expected per-tick budget ({}); inject_fn may be over-publishing",
            self.events.len(),
            self.agents.len() * 2,
        );
        self.tick += 1;
        self.events.clear();
    }

    pub fn snapshot(&self) -> WorldSnapshot {
        WorldSnapshot {
            tick: self.tick,
            agents: self.agents.clone(),
            events: self.events.clone(),
            variables: self.variables.clone(),
        }
    }
}

impl Default for WorldState {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldSnapshot {
    pub tick: u32,
    pub agents: HashMap<Uuid, AgentSnapshot>,
    pub events: Vec<Event>,
    pub variables: HashMap<String, f32>,
}

impl WorldSnapshot {
    /// Get a variable value from the world snapshot.
    ///
    /// This provides the same interface as `WorldState::get_variable()`.
    ///
    /// # Arguments
    /// * `key` - Variable name to lookup
    ///
    /// # Returns
    /// * `Some(value)` if the variable exists
    /// * `None` if the variable does not exist
    pub fn get_variable(&self, key: &str) -> Option<f32> {
        self.variables.get(key).copied()
    }
}

pub type InjectFn = std::sync::Arc<dyn Fn(u32, &mut WorldState) + Send + Sync>;

/// Configuration for simulation execution.
///
/// Defines tick limits, parallelism level, and an optional injection function
/// for external control of world state (the "God's-eye" mechanism).
///
/// The injection function allows external code to modify the simulation state
/// at each tick, enabling "what-if" scenarios or external control systems.
///
/// # Fields
///
/// * `max_ticks` - Maximum number of simulation ticks to run before stopping
/// * `parallelism` - Max concurrent LLM calls per tick (used by `SimEngine::run`)
/// * `inject_fn` - Optional function called at each tick to modify world state
///
/// # Memory characteristics
///
/// `SimEngine::run` holds all tick snapshots in memory for the full duration
/// of the simulation. Memory usage is approximately
/// `O(max_ticks * agent_count * snapshot_size)`. For large simulations
/// (e.g. `max_ticks > 1000` with large pools), monitor heap usage and
/// consider reducing `max_ticks` or snapshotting to disk.
///
/// # Note on `Clone`
///
/// `SimConfig` implements `Clone` because the injection function is wrapped in
/// `Arc<dyn Fn>`, which is shareable across threads.
///
/// # Example
///
/// ```ignore
/// let config = SimConfig::new(100, 8)
///     .with_inject_fn(|tick, world| {
///         if tick == 50 {
///             world.inject_variable("halfway".to_string(), 1.0);
///         }
///     });
/// ```
#[derive(Clone)]
pub struct SimConfig {
    pub max_ticks: u32,
    /// Maximum number of concurrent LLM calls per tick.
    /// Controls `stream::buffered(parallelism)` in `SimEngine::run`.
    /// Set to 1 to execute agents sequentially; higher values increase
    /// throughput at the cost of additional concurrent HTTP connections.
    pub parallelism: usize,
    pub inject_fn: Option<InjectFn>,
}

impl SimConfig {
    /// Create a new `SimConfig` with the specified tick limit and parallelism.
    ///
    /// The injection function is not set; use `with_inject_fn()` to add one.
    ///
    /// # Arguments
    ///
    /// * `max_ticks` - Maximum number of simulation ticks to run
    /// * `parallelism` - Number of threads for parallel agent execution
    ///
    /// # Example
    ///
    /// ```
    /// let config = SimConfig::new(100, 8);
    /// ```
    pub fn new(max_ticks: u32, parallelism: usize) -> Self {
        Self { max_ticks, parallelism, inject_fn: None }
    }

    /// Register an injection function to modify world state at each tick.
    ///
    /// The injection function is called by the simulation engine at each tick
    /// with the current tick number and a mutable reference to the `WorldState`.
    /// This allows external code to inject or modify world variables based on
    /// the simulation progress (the "God's-eye" mechanism).
    ///
    /// # Arguments
    ///
    /// * `inject_fn` - A function that takes (tick: u32, world: &mut WorldState)
    ///
    /// # Example
    ///
    /// ```
    /// let config = SimConfig::new(100, 4)
    ///     .with_inject_fn(|tick, world| {
    ///         // Increase temperature every 10 ticks
    ///         if tick % 10 == 0 {
    ///             let current_temp = world.get_variable("temp").unwrap_or(20.0);
    ///             world.inject_variable("temp".to_string(), current_temp + 1.0);
    ///         }
    ///     });
    /// ```
    pub fn with_inject_fn<F>(mut self, inject_fn: F) -> Self
    where
        F: Fn(u32, &mut WorldState) + Send + Sync + 'static,
    {
        self.inject_fn = Some(std::sync::Arc::new(inject_fn));
        self
    }
}

impl std::fmt::Debug for SimConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimConfig")
            .field("max_ticks", &self.max_ticks)
            .field("parallelism", &self.parallelism)
            .field("inject_fn", &self.inject_fn.as_ref().map(|_| "<function>"))
            .finish()
    }
}

impl Default for SimConfig {
    fn default() -> Self {
        Self { max_ticks: 50, parallelism: 8, inject_fn: None }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub id: Uuid,
    pub history: Vec<WorldSnapshot>,
}

impl SimulationResult {
    /// Returns a reference to the last snapshot in history, i.e. the final world state.
    pub fn final_snapshot(&self) -> Option<&WorldSnapshot> {
        self.history.last()
    }
}

/// Callback type for snapshot hooks registered with `SimEngine`.
/// Each hook is called once per tick with a clone of the tick's snapshot.
pub type SnapshotHook = Arc<dyn Fn(WorldSnapshot) + Send + Sync>;

pub struct SimEngine {
    config: SimConfig,
    snapshot_tx: broadcast::Sender<WorldSnapshot>,
    snapshot_history: Arc<Mutex<Vec<WorldSnapshot>>>,
    /// Registered snapshot hooks (e.g. TickBuffer adapters for HTTP streaming).
    snapshot_hooks: Vec<SnapshotHook>,
}

impl SimEngine {
    pub fn new(config: SimConfig) -> Self {
        // Fixed capacity of 64: gives slow receivers a short grace window before
        // RecvError::Lagged. History replay via subscribe_with_history() covers
        // ticks beyond the 64-slot window.
        let (snapshot_tx, _snapshot_rx) = broadcast::channel(64);
        Self {
            config,
            snapshot_tx,
            snapshot_history: Arc::new(Mutex::new(Vec::new())),
            snapshot_hooks: Vec::new(),
        }
    }

    /// Register a snapshot hook called once per tick during `run()`.
    /// Use `StreamAdapter::as_hook()` to wire a `TickBuffer` for HTTP streaming.
    pub fn register_snapshot_hook(&mut self, hook: SnapshotHook) {
        self.snapshot_hooks.push(hook);
    }

    pub fn config(&self) -> &SimConfig {
        &self.config
    }

    pub fn subscribe(&self) -> broadcast::Receiver<WorldSnapshot> {
        self.snapshot_tx.subscribe()
    }

    /// Subscribe to live tick snapshots and get a handle to all snapshots produced so far.
    ///
    /// The returned `Arc<Mutex<Vec<WorldSnapshot>>>` is populated tick-by-tick during `run()`.
    /// Callers who start listening after `run()` has already begun can drain the vec to replay
    /// any missed ticks, then consume the `Receiver` for subsequent ticks.
    ///
    /// # Deduplication contract
    /// A snapshot for tick `n` may appear in both the history `Vec` (if the caller subscribes
    /// after tick `n` was broadcast) **and** in the `Receiver` (if the caller's receiver was
    /// created before tick `n` was sent). Callers must deduplicate by `WorldSnapshot::tick`
    /// when combining replay history with live receiver output.
    pub fn subscribe_with_history(
        &self,
    ) -> (broadcast::Receiver<WorldSnapshot>, Arc<Mutex<Vec<WorldSnapshot>>>) {
        (self.snapshot_tx.subscribe(), Arc::clone(&self.snapshot_history))
    }

    pub async fn run<L: crate::llm::LlmClient>(
        &self,
        pool: &mut crate::agent::AgentPool,
        // TODO(graph-context): pass per-agent subgraph slices once Agent::prepare_action
        // accepts a graph reference. Tracked: _graph param intentionally kept so callers
        // do not need an API change when the feature lands.
        _graph: &crate::graph::KnowledgeGraph,
        llm: &L,
    ) -> crate::error::Result<SimulationResult> {
        use futures::stream::{self, StreamExt};

        self.snapshot_history.lock().clear();
        let mut world = WorldState::new();

        // Seed agent snapshots into world state
        for agent in pool.iter() {
            world.add_agent_snapshot(
                agent.id,
                AgentSnapshot {
                    id: agent.id,
                    name: agent.persona.name.clone(),
                    state: format!("{:?}", agent.state),
                },
            );
        }

        for _ in 0..self.config.max_ticks {
            world.advance_tick();

            // Phase 1: prepare actions concurrently (immutable reads + LLM calls).
            // stream::buffered drives at most `parallelism` futures simultaneously,
            // giving real throughput gains when agent steps are LLM-bound.
            let actions: Vec<crate::error::Result<crate::sim::Action>> =
                stream::iter(pool.agents.iter())
                    .map(|agent| agent.prepare_action(&world, llm))
                    .buffered(self.config.parallelism)
                    .collect()
                    .await;

            // Phase 2: commit results sequentially (mutable writes + world state).
            for (agent, action_result) in pool.agents.iter_mut().zip(actions.into_iter()) {
                let action = action_result?;
                world.apply(agent.id, action.clone());
                agent.commit_action(&action);
                if let Some(snap) = world.agents.get_mut(&agent.id) {
                    snap.state = format!("{:?}", agent.state);
                }
            }

            // Apply God's-eye injection if configured
            if let Some(ref inject) = self.config.inject_fn {
                inject(world.tick, &mut world);
            }

            let snapshot = world.snapshot();
            // Broadcast to live subscribers (RecvError::Lagged signals gap to slow consumers)
            let _ = self.snapshot_tx.send(snapshot.clone());
            // Call registered hooks (e.g. TickBuffer adapters for HTTP streaming — 3A)
            for hook in &self.snapshot_hooks {
                hook(snapshot.clone());
            }
            // snapshot_history is the single canonical in-memory store (6A)
            self.snapshot_history.lock().push(snapshot);
        }

        // Clone history from canonical store; avoids a local Vec running in parallel (6A)
        let history = self.snapshot_history.lock().clone();
        Ok(SimulationResult { id: Uuid::new_v4(), history })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_state_creation() {
        let world = WorldState::new();
        assert_eq!(world.tick, 0);
        assert!(world.agents.is_empty());
        assert!(world.events.is_empty());
    }

    #[test]
    fn test_world_state_advance_tick() {
        let mut world = WorldState::new();
        world.advance_tick();
        assert_eq!(world.tick, 1);
    }

    #[test]
    fn test_world_state_variables() {
        let mut world = WorldState::new();
        world.inject_variable("temperature".to_string(), 25.5);
        assert_eq!(world.get_variable("temperature"), Some(25.5));
    }

    #[test]
    fn test_world_snapshot() {
        let world = WorldState::new();
        let snapshot = world.snapshot();
        assert_eq!(snapshot.tick, world.tick);
    }

    #[test]
    fn test_sim_engine_creation() {
        let config = SimConfig { max_ticks: 100, parallelism: 4, inject_fn: None };
        let engine = SimEngine::new(config);
        assert_eq!(engine.config().max_ticks, 100);
    }

    #[test]
    fn test_world_state_apply() {
        let mut world = WorldState::new();
        let agent_id = Uuid::new_v4();
        world.apply(agent_id, Action::Think("pondering".to_string()));
        assert_eq!(world.events.len(), 1);
        assert_eq!(world.events[0].agent_id, agent_id);
    }

    #[test]
    fn test_world_state_apply_at_deterministic() {
        let mut world = WorldState::new();
        let agent_id = Uuid::new_v4();
        let ts = chrono::DateTime::from_timestamp(0, 0).unwrap();
        world.apply_at(agent_id, Action::Speak("hello".to_string()), ts);
        assert_eq!(world.events.len(), 1);
        assert_eq!(world.events[0].timestamp, ts);
    }

    #[test]
    fn test_sim_engine_subscribe() {
        let config = SimConfig::default();
        let engine = SimEngine::new(config);
        let rx = engine.subscribe();
        assert_eq!(rx.len(), 0); // channel is empty
    }

    #[test]
    fn test_subscribe_with_history_returns_shared_arc() {
        let config = SimConfig::default();
        let engine = SimEngine::new(config);
        let (_rx, history) = engine.subscribe_with_history();
        assert_eq!(history.lock().len(), 0);
        // Simulate a tick being pushed (as run() would do)
        let world = WorldState::new();
        history.lock().push(world.snapshot());
        assert_eq!(history.lock().len(), 1);
    }

    #[test]
    fn test_sim_config_with_inject_fn() {
        let inject: InjectFn = std::sync::Arc::new(|tick, world| {
            world.inject_variable("tick".to_string(), tick as f32);
        });
        let config = SimConfig { max_ticks: 10, parallelism: 2, inject_fn: Some(inject) };
        assert_eq!(config.max_ticks, 10);
    }

    #[test]
    fn test_sim_config_new_constructor() {
        let config = SimConfig::new(100, 4);
        assert_eq!(config.max_ticks, 100);
        assert_eq!(config.parallelism, 4);
        assert!(config.inject_fn.is_none());
    }

    #[test]
    fn test_sim_config_with_inject_fn_builder() {
        let config = SimConfig::new(100, 4).with_inject_fn(|tick, world| {
            if tick == 5 {
                world.inject_variable("test_var".to_string(), 42.0);
            }
        });

        assert_eq!(config.max_ticks, 100);
        assert_eq!(config.parallelism, 4);
        assert!(config.inject_fn.is_some());
    }

    #[test]
    fn test_sim_config_builder_chain() {
        let config = SimConfig::new(200, 8).with_inject_fn(|tick, world| {
            world.inject_variable("tick_count".to_string(), tick as f32);
        });

        assert_eq!(config.max_ticks, 200);
        assert_eq!(config.parallelism, 8);
        assert!(config.inject_fn.is_some());
    }

    #[test]
    fn test_world_snapshot_get_variable() {
        let mut world = WorldState::new();
        world.inject_variable("temperature".to_string(), 25.5);
        world.inject_variable("humidity".to_string(), 65.0);

        let snapshot = world.snapshot();

        // Test existing variables
        assert_eq!(snapshot.get_variable("temperature"), Some(25.5));
        assert_eq!(snapshot.get_variable("humidity"), Some(65.0));

        // Test non-existent variable
        assert_eq!(snapshot.get_variable("nonexistent"), None);

        // Test that variables are properly cloned
        world.inject_variable("temperature".to_string(), 30.0); // Modify original
        assert_eq!(snapshot.get_variable("temperature"), Some(25.5)); // Snapshot unchanged
    }

    #[test]
    fn test_world_snapshot_preserves_variables() {
        let mut world = WorldState::new();
        world.inject_variable("test".to_string(), 42.0);

        let snapshot = world.snapshot();

        // Verify snapshot contains variables
        assert_eq!(snapshot.get_variable("test"), Some(42.0));
        assert_eq!(snapshot.variables.len(), 1);

        // Verify variables are accessible via get_variable
        assert_eq!(snapshot.get_variable("test"), world.get_variable("test"));
    }

    #[test]
    fn test_inject_fn_variable_modification() {
        // Test that the injection function can actually modify world variables
        let mut world = WorldState::new();
        world.inject_variable("counter".to_string(), 0.0);

        let config = SimConfig::new(1, 1).with_inject_fn(|tick, world| {
            let current = world.get_variable("counter").unwrap_or(0.0);
            world.inject_variable("counter".to_string(), current + tick as f32);
        });

        // Manually call the injection function
        if let Some(ref inject) = config.inject_fn {
            inject(5, &mut world);
        }

        assert_eq!(world.get_variable("counter"), Some(5.0));
    }

    #[tokio::test]
    async fn test_sim_engine_runs_multiple_agents() {
        // 9A: verify SimEngine::run executes all agents each tick and collects
        // their actions into the world snapshot. Uses a mock LLM.
        use crate::agent::{Agent, AgentPool, Persona};
        use crate::error::Result;
        use crate::llm::LlmClient;
        use async_trait::async_trait;
        use std::pin::Pin;

        struct MockLlm;
        #[async_trait]
        impl LlmClient for MockLlm {
            async fn complete(&self, _: &str) -> Result<String> {
                Ok("Speak(hello from mock)".to_string())
            }
            async fn complete_json<T: serde::de::DeserializeOwned>(&self, _: &str) -> Result<T> {
                Err(crate::error::TeriError::Llm("not used".into()))
            }
            async fn stream(
                &self,
                _: &str,
            ) -> Result<Pin<Box<dyn futures::Stream<Item = Result<String>> + Send>>> {
                Err(crate::error::TeriError::Llm("not used".into()))
            }
        }

        let mut pool = AgentPool::new();
        for i in 0..3 {
            let persona = Persona {
                name: format!("Agent-{}", i),
                background: "Test agent".to_string(),
                traits: vec!["test".to_string()],
                role: "tester".to_string(),
            };
            pool.add_agent(Agent::new(persona));
        }

        let config = SimConfig::new(2, 3); // 2 ticks, 3 concurrent (all agents in parallel)
        let engine = SimEngine::new(config);
        let graph = crate::graph::KnowledgeGraph::new();
        let llm = MockLlm;

        let result = engine.run(&mut pool, &graph, &llm).await.expect("run failed");

        // 2 ticks recorded
        assert_eq!(result.history.len(), 2);
        // Each tick has 3 events (one per agent)
        for snapshot in &result.history {
            assert_eq!(snapshot.events.len(), 3, "expected 3 events at tick {}", snapshot.tick);
        }
        // Tick numbers increment correctly
        assert_eq!(result.history[0].tick, 1);
        assert_eq!(result.history[1].tick, 2);
        // final_snapshot convenience method works
        assert_eq!(result.final_snapshot().unwrap().tick, 2);
    }

    #[tokio::test]
    async fn test_integration_small_agent_pool() {
        // 10A: integration test that actually calls engine.run() with inject_fn,
        // verifies inject_fn variables appear in snapshots and history is complete.
        use crate::agent::{Agent, AgentPool, Persona};
        use crate::error::Result;
        use crate::llm::LlmClient;
        use async_trait::async_trait;
        use std::pin::Pin;

        struct MockLlm;
        #[async_trait]
        impl LlmClient for MockLlm {
            async fn complete(&self, _: &str) -> Result<String> {
                Ok("Think(exploring)".to_string())
            }
            async fn complete_json<T: serde::de::DeserializeOwned>(&self, _: &str) -> Result<T> {
                Err(crate::error::TeriError::Llm("not used".into()))
            }
            async fn stream(
                &self,
                _: &str,
            ) -> Result<Pin<Box<dyn futures::Stream<Item = Result<String>> + Send>>> {
                Err(crate::error::TeriError::Llm("not used".into()))
            }
        }

        let mut pool = AgentPool::new();
        for i in 0..2 {
            let persona = Persona {
                name: format!("TestAgent-{}", i),
                background: format!("Test agent {}", i),
                traits: vec!["curious".to_string()],
                role: "explorer".to_string(),
            };
            pool.add_agent(Agent::new(persona));
        }

        let config = SimConfig::new(3, 2).with_inject_fn(|tick, world| {
            world.inject_variable("sim_tick".to_string(), tick as f32);
            world.inject_variable("pressure".to_string(), 1000.0 + (tick as f32 * 5.0));
        });

        let engine = SimEngine::new(config);
        let graph = crate::graph::KnowledgeGraph::new();
        let llm = MockLlm;

        let result = engine.run(&mut pool, &graph, &llm).await.expect("run failed");

        // 3 ticks in history
        assert_eq!(result.history.len(), 3);

        // inject_fn variables present in each snapshot
        for (i, snapshot) in result.history.iter().enumerate() {
            let expected_tick = (i + 1) as f32;
            assert_eq!(
                snapshot.get_variable("sim_tick"),
                Some(expected_tick),
                "sim_tick wrong at history index {i}"
            );
            assert_eq!(
                snapshot.get_variable("pressure"),
                Some(1000.0 + expected_tick * 5.0),
                "pressure wrong at history index {i}"
            );
        }

        // 2 events per tick (one per agent)
        for snapshot in &result.history {
            assert_eq!(snapshot.events.len(), 2);
        }
    }

    #[tokio::test]
    async fn test_sim_engine_run_basic_with_broadcast() {
        // 11A + 12A: thorough test of engine.run() — history, event count, tick order,
        // and broadcast receiver receives all snapshots in order.
        use crate::agent::{Agent, AgentPool, Persona};
        use crate::error::Result;
        use crate::llm::LlmClient;
        use async_trait::async_trait;
        use std::pin::Pin;

        struct MockLlm;
        #[async_trait]
        impl LlmClient for MockLlm {
            async fn complete(&self, _: &str) -> Result<String> {
                Ok("Observe(the room)".to_string())
            }
            async fn complete_json<T: serde::de::DeserializeOwned>(&self, _: &str) -> Result<T> {
                Err(crate::error::TeriError::Llm("not used".into()))
            }
            async fn stream(
                &self,
                _: &str,
            ) -> Result<Pin<Box<dyn futures::Stream<Item = Result<String>> + Send>>> {
                Err(crate::error::TeriError::Llm("not used".into()))
            }
        }

        const TICKS: u32 = 4;
        const AGENTS: usize = 2;

        let mut pool = AgentPool::new();
        for i in 0..AGENTS {
            pool.add_agent(Agent::new(Persona {
                name: format!("Bot-{i}"),
                background: "test".into(),
                traits: vec!["test".into()],
                role: "observer".into(),
            }));
        }

        let config = SimConfig::new(TICKS, AGENTS);
        let engine = SimEngine::new(config);

        // 12A: subscribe BEFORE run so receiver captures all ticks
        let mut rx = engine.subscribe();

        let graph = crate::graph::KnowledgeGraph::new();
        let llm = MockLlm;
        let result = engine.run(&mut pool, &graph, &llm).await.expect("run failed");

        // History correctness
        assert_eq!(result.history.len(), TICKS as usize);
        for (i, snap) in result.history.iter().enumerate() {
            assert_eq!(snap.tick, (i + 1) as u32);
            assert_eq!(snap.events.len(), AGENTS, "tick {} must have {} events", snap.tick, AGENTS);
        }

        // Broadcast receiver received all TICKS snapshots in order
        let mut received = Vec::new();
        while let Ok(snap) = rx.try_recv() {
            received.push(snap);
        }
        assert_eq!(received.len(), TICKS as usize, "broadcast delivered wrong number of snapshots");
        for (i, snap) in received.iter().enumerate() {
            assert_eq!(snap.tick, (i + 1) as u32, "broadcast tick order wrong at index {i}");
        }
    }
}
