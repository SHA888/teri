use serde::{Deserialize, Serialize};
use std::collections::HashMap;
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

#[derive(Debug, Clone)]
pub struct SimConfig {
    pub max_ticks: u32,
    pub parallelism: usize,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self { max_ticks: 50, parallelism: 8 }
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
}

impl SimEngine {
    pub fn new(config: SimConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &SimConfig {
        &self.config
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
        let config = SimConfig { max_ticks: 100, parallelism: 4 };
        let engine = SimEngine::new(config);
        assert_eq!(engine.config().max_ticks, 100);
    }
}
