use crate::error::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Persona {
    pub name: String,
    pub background: String,
    pub traits: Vec<String>,
    pub role: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AgentState {
    Idle,
    Thinking,
    Acting,
    Observing,
    Communicating,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub timestamp: chrono::DateTime<Utc>,
    pub content: String,
    pub importance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMemory {
    pub short_term: VecDeque<MemoryEntry>,
    pub short_term_capacity: usize,
}

impl AgentMemory {
    pub fn new(capacity: usize) -> Self {
        Self {
            short_term: VecDeque::with_capacity(capacity),
            short_term_capacity: capacity,
        }
    }

    pub fn add_memory(&mut self, entry: MemoryEntry) {
        if self.short_term.len() >= self.short_term_capacity {
            self.short_term.pop_front();
        }
        self.short_term.push_back(entry);
    }

    pub fn get_recent(&self, limit: usize) -> Vec<&MemoryEntry> {
        self.short_term
            .iter()
            .rev()
            .take(limit)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    pub fn clear(&mut self) {
        self.short_term.clear();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub id: Uuid,
    pub persona: Persona,
    pub memory: AgentMemory,
    pub state: AgentState,
}

impl Agent {
    pub fn new(persona: Persona) -> Self {
        Self {
            id: Uuid::new_v4(),
            persona,
            memory: AgentMemory::new(50),
            state: AgentState::Idle,
        }
    }

    pub fn add_memory(&mut self, content: String, importance: f32) {
        let entry = MemoryEntry {
            timestamp: Utc::now(),
            content,
            importance,
        };
        self.memory.add_memory(entry);
    }

    pub fn set_state(&mut self, state: AgentState) {
        self.state = state;
    }
}

pub struct AgentPool {
    pub agents: Vec<Agent>,
}

impl AgentPool {
    pub fn new() -> Self {
        Self {
            agents: Vec::new(),
        }
    }

    pub fn add_agent(&mut self, agent: Agent) {
        self.agents.push(agent);
    }

    pub fn get(&self, id: Uuid) -> Option<&Agent> {
        self.agents.iter().find(|a| a.id == id)
    }

    pub fn get_mut(&mut self, id: Uuid) -> Option<&mut Agent> {
        self.agents.iter_mut().find(|a| a.id == id)
    }

    pub fn len(&self) -> usize {
        self.agents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.agents.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Agent> {
        self.agents.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Agent> {
        self.agents.iter_mut()
    }
}

impl Default for AgentPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_creation() {
        let persona = Persona {
            name: "Alice".to_string(),
            background: "A curious researcher".to_string(),
            traits: vec!["analytical".to_string(), "creative".to_string()],
            role: "Analyst".to_string(),
        };

        let agent = Agent::new(persona.clone());
        assert_eq!(agent.persona.name, "Alice");
        assert_eq!(agent.state, AgentState::Idle);
    }

    #[test]
    fn test_agent_memory() {
        let persona = Persona {
            name: "Alice".to_string(),
            background: "A curious researcher".to_string(),
            traits: vec!["analytical".to_string()],
            role: "Analyst".to_string(),
        };

        let mut agent = Agent::new(persona);
        agent.add_memory("First memory".to_string(), 0.8);
        agent.add_memory("Second memory".to_string(), 0.9);

        assert_eq!(agent.memory.short_term.len(), 2);
    }

    #[test]
    fn test_agent_pool() {
        let mut pool = AgentPool::new();
        let persona = Persona {
            name: "Alice".to_string(),
            background: "A curious researcher".to_string(),
            traits: vec!["analytical".to_string()],
            role: "Analyst".to_string(),
        };

        let agent = Agent::new(persona);
        let agent_id = agent.id;
        pool.add_agent(agent);

        assert_eq!(pool.len(), 1);
        assert!(pool.get(agent_id).is_some());
    }

    #[test]
    fn test_agent_state_change() {
        let persona = Persona {
            name: "Alice".to_string(),
            background: "A curious researcher".to_string(),
            traits: vec!["analytical".to_string()],
            role: "Analyst".to_string(),
        };

        let mut agent = Agent::new(persona);
        assert_eq!(agent.state, AgentState::Idle);

        agent.set_state(AgentState::Thinking);
        assert_eq!(agent.state, AgentState::Thinking);
    }
}
