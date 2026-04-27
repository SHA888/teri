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

pub type InjectFn = std::sync::Arc<dyn Fn(u32, &mut WorldState) + Send + Sync>;

#[derive(Clone)]
pub struct SimConfig {
    pub max_ticks: u32,
    /// Reserved for future parallel async agent execution (e.g. scoped tokio tasks).
    /// Currently unused — `SimEngine::run` executes agents sequentially.
    pub parallelism: usize,
    pub inject_fn: Option<InjectFn>,
}

impl std::fmt::Debug for SimConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimConfig")
            .field("max_ticks", &self.max_ticks)
            .field("parallelism", &self.parallelism)
            .field("inject_fn", &self.inject_fn.is_some())
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
}
