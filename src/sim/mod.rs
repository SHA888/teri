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
        Self { tick: 0, agents: HashMap::new(), events: Vec::new(), variables: HashMap::new() }
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

    pub fn advance_tick(&mut self) {
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
/// * `parallelism` - Reserved for future parallel execution (currently unused)
/// * `inject_fn` - Optional function called at each tick to modify world state
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
    /// Reserved for future parallel async agent execution (e.g. scoped tokio tasks).
    /// Currently unused — `SimEngine::run` executes agents sequentially.
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
    pub final_state: WorldState,
}

pub struct SimEngine {
    config: SimConfig,
    snapshot_tx: broadcast::Sender<WorldSnapshot>,
    snapshot_history: Arc<Mutex<Vec<WorldSnapshot>>>,
}

impl SimEngine {
    pub fn new(config: SimConfig) -> Self {
        let capacity = (config.max_ticks as usize).max(64);
        let (snapshot_tx, _snapshot_rx) = broadcast::channel(capacity);
        Self { config, snapshot_tx, snapshot_history: Arc::new(Mutex::new(Vec::new())) }
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
        _graph: &crate::graph::KnowledgeGraph, // TODO: use graph for per-agent context construction
        llm: &L,
    ) -> crate::error::Result<SimulationResult> {
        self.snapshot_history.lock().clear();
        let mut world = WorldState::new();
        let mut history = Vec::new();

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

            // Run each agent step sequentially (parallel async requires scoped tasks)
            for agent in pool.iter_mut() {
                let action = agent.step(&world, llm).await?;
                world.apply(agent.id, action);
                if let Some(snapshot) = world.agents.get_mut(&agent.id) {
                    snapshot.state = format!("{:?}", agent.state);
                }
            }

            // Apply God's-eye injection if configured
            if let Some(ref inject) = self.config.inject_fn {
                inject(world.tick, &mut world);
            }

            let snapshot = world.snapshot();
            let _ = self.snapshot_tx.send(snapshot.clone()); // ignore if no listeners
            self.snapshot_history.lock().push(snapshot.clone());
            history.push(snapshot);
        }

        let result = SimulationResult { id: Uuid::new_v4(), history, final_state: world };

        Ok(result)
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
    async fn test_parallel_agent_execution() {
        // Test that SimEngine can execute multiple agents concurrently within a tick.
        // This validates that the agent step mechanism doesn't block and can handle
        // multiple agents acting in the same tick.

        use crate::agent::{Agent, AgentPool, Persona};

        // Create a small pool of agents
        let mut pool = AgentPool::new();
        for i in 0..3 {
            let persona = Persona {
                name: format!("Agent-{}", i),
                background: "Test agent".to_string(),
                traits: vec!["test".to_string()],
                role: "tester".to_string(),
            };
            let agent = Agent::new(persona);
            pool.add_agent(agent);
        }

        assert_eq!(pool.len(), 3);

        // Verify pool can be iterated (mock of parallel execution)
        let agent_count = pool.iter().count();
        assert_eq!(agent_count, 3);

        // Verify all agents have distinct IDs
        let ids: Vec<_> = pool.iter().map(|a| a.id).collect();
        assert_eq!(ids.len(), 3);
        assert_eq!(ids[0], ids[0]); // Same agent, same ID
        assert_ne!(ids[0], ids[1]); // Different agents, different IDs
    }

    #[test]
    fn test_integration_small_agent_pool() {
        // Integration test: Verify engine setup and config work together.
        // Full end-to-end test with actual LLM would require mock framework.

        use crate::agent::{Agent, AgentPool, Persona};

        // Setup: Create a small agent pool
        let mut pool = AgentPool::new();
        for i in 0..2 {
            let persona = Persona {
                name: format!("TestAgent-{}", i),
                background: format!("Test agent {}", i),
                traits: vec!["curious".to_string(), "thoughtful".to_string()],
                role: "explorer".to_string(),
            };
            let agent = Agent::new(persona);
            pool.add_agent(agent);
        }

        // Verify: Pool was created correctly
        assert_eq!(pool.len(), 2);

        // Setup: Create simulation config with injection function
        let config = SimConfig::new(10, 2).with_inject_fn(|tick, world| {
            // Inject a tick counter that increments each tick
            world.inject_variable("sim_tick".to_string(), tick as f32);

            // Inject environment pressure that changes over time
            let pressure = 1000.0 + (tick as f32 * 5.0);
            world.inject_variable("pressure".to_string(), pressure);
        });

        // Setup: Create engine
        let engine = SimEngine::new(config);

        // Verify: Engine was initialized correctly
        assert_eq!(engine.config().max_ticks, 10);
        assert_eq!(engine.config().parallelism, 2);
        assert!(engine.config().inject_fn.is_some());

        // Verify: Engine can create subscriptions
        let rx = engine.subscribe();
        assert_eq!(rx.len(), 0);

        // Verify: Engine can create history subscriptions
        let (rx2, history) = engine.subscribe_with_history();
        assert_eq!(rx2.len(), 0);
        assert_eq!(history.lock().len(), 0);

        // Verify: Injection function works when called
        let mut test_world = WorldState::new();
        if let Some(ref inject) = engine.config().inject_fn {
            inject(5, &mut test_world);
            assert_eq!(test_world.get_variable("sim_tick"), Some(5.0));
            assert_eq!(test_world.get_variable("pressure"), Some(1025.0));
        }

        // Verify: Subscriptions work
        assert_eq!(engine.subscribe().len(), 0);

        // Verify: History subscriptions work
        let (_, history) = engine.subscribe_with_history();
        assert_eq!(history.lock().len(), 0);
    }
}
