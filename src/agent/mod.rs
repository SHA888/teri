use crate::error::{Result, TeriError};
use crate::graph::{Entity, KnowledgeGraph};
use crate::llm::LlmClient;
use crate::sim::{Action, WorldState};
use chrono::Utc;
use minijinja::{Environment, context};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::RwLock;
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
        Self { short_term: VecDeque::with_capacity(capacity), short_term_capacity: capacity }
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
        let entry = MemoryEntry { timestamp: Utc::now(), content, importance };
        self.memory.add_memory(entry);
    }

    pub fn set_state(&mut self, state: AgentState) {
        self.state = state;
    }

    /// Execute one step of the agent's decision-making process
    pub async fn step<L: LlmClient>(&mut self, world: &WorldState, llm: &L) -> Result<Action> {
        // Set state to Thinking
        self.set_state(AgentState::Thinking);

        // Retrieve relevant memories
        let relevant_memories = self.retrieve_relevant_memories(world);

        // Construct context from world state + memories
        let context = self.construct_context(world, &relevant_memories);

        // Set state to Acting
        self.set_state(AgentState::Acting);

        // Generate action using LLM with fallback
        let action = self.generate_action_with_fallback(&context, llm).await?;

        // Parse and validate action
        let validated_action = self.parse_and_validate_action(&action)?;

        // Store action in memory
        self.store_action_in_memory(&validated_action);

        // Return to Idle state
        self.set_state(AgentState::Idle);

        Ok(validated_action)
    }

    /// Retrieve relevant memories based on current world state
    fn retrieve_relevant_memories(&self, _world: &WorldState) -> Vec<&MemoryEntry> {
        // Get recent memories (simple implementation - could be enhanced with relevance scoring)
        self.memory.get_recent(10)
    }

    /// Construct context string from world state and memories
    fn construct_context(&self, world: &WorldState, memories: &[&MemoryEntry]) -> String {
        let mut context = format!(
            "Agent: {}\nRole: {}\nState: {:?}\n\n",
            self.persona.name, self.persona.role, self.state
        );

        context.push_str(&format!("World Tick: {}\n\n", world.tick));

        // Add recent events with agent names
        if !world.events.is_empty() {
            context.push_str("Recent Events:\n");
            for event in world.events.iter().rev().take(5) {
                let agent_name = world
                    .agents
                    .get(&event.agent_id)
                    .map(|snapshot| snapshot.name.as_str())
                    .unwrap_or("Unknown Agent");
                context.push_str(&format!("- {}: {}\n", agent_name, event.action));
            }
            context.push('\n');
        }

        // Add memories
        if !memories.is_empty() {
            context.push_str("Relevant Memories:\n");
            for memory in memories {
                context.push_str(&format!("- {}\n", memory.content));
            }
            context.push('\n');
        }

        // Add world variables
        if !world.variables.is_empty() {
            context.push_str("World State:\n");
            for (key, value) in &world.variables {
                context.push_str(&format!("- {}: {:.2}\n", key, value));
            }
        }

        context
    }

    /// Generate action using LLM with context and fallback
    async fn generate_action_with_fallback<L: LlmClient>(
        &self,
        context: &str,
        llm: &L,
    ) -> Result<String> {
        // Try to generate action
        match self.generate_action(context, llm).await {
            Ok(action) => Ok(action),
            Err(_) => {
                // Fallback to a simple thinking action
                Ok("Think(I need to consider my next move carefully)".to_string())
            }
        }
    }

    /// Generate action using LLM with context
    async fn generate_action<L: LlmClient>(&self, context: &str, llm: &L) -> Result<String> {
        let generator = ActionGenerator::new();
        let prompt = generator.generate_prompt(self, context)?;

        llm.complete(&prompt).await
    }

    /// Parse and validate the action string with robust parsing
    fn parse_and_validate_action(&self, action_str: &str) -> Result<Action> {
        let action_str = action_str.trim();

        // Find the first '(' and the last ')' to handle nested parentheses
        if let Some(paren_start) = action_str.find('(')
            && let Some(paren_end) = action_str.rfind(')')
            && paren_end > paren_start
        {
            let action_type = &action_str[..paren_start];
            let content = &action_str[paren_start + 1..paren_end];

            return match action_type.trim() {
                "Speak" => Ok(Action::Speak(content.trim().to_string())),
                "Move" => Ok(Action::Move(content.trim().to_string())),
                "Interact" => Ok(Action::Interact(content.trim().to_string())),
                "Observe" => Ok(Action::Observe(content.trim().to_string())),
                "Think" => Ok(Action::Think(content.trim().to_string())),
                _ => Err(TeriError::Agent(format!("Unknown action type: {}", action_type))),
            };
        }

        Err(TeriError::Agent(format!("Invalid action format: {}", action_str)))
    }

    /// Store the executed action in memory with dynamic importance
    fn store_action_in_memory(&mut self, action: &Action) {
        let (memory_content, importance) = match action {
            Action::Speak(content) => {
                let importance = if content.len() > 100 { 0.8 } else { 0.6 };
                (format!("Spoke: {}", content), importance)
            }
            Action::Move(location) => (format!("Moved to: {}", location), 0.7),
            Action::Interact(target) => (format!("Interacted with: {}", target), 0.8),
            Action::Observe(target) => (format!("Observed: {}", target), 0.5),
            Action::Think(content) => {
                let importance = if content.contains("plan") || content.contains("strategy") {
                    0.9
                } else {
                    0.4
                };
                (format!("Thought: {}", content), importance)
            }
        };

        self.add_memory(memory_content, importance);
    }
}

#[derive(Debug, Clone)]
/// A pool of agents with shared group memory.
///
/// # Clone Behavior
///
/// Cloning an AgentPool creates a new instance that shares the same group memory
/// through `Arc<RwLock<>>`. This means both pools will share the same group memory
/// data, but have separate agent vectors. This is the desired behavior for shared
/// memory scenarios, but be aware that modifications to group memory will be visible
/// to all cloned instances.
pub struct AgentPool {
    pub agents: Vec<Agent>,
    pub group_memory: Arc<RwLock<Vec<MemoryEntry>>>,
}

impl AgentPool {
    pub fn new() -> Self {
        Self { agents: Vec::new(), group_memory: Arc::new(RwLock::new(Vec::new())) }
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

    /// Spawn N unique agents using personas generated from the knowledge graph
    pub async fn spawn<L: LlmClient>(n: usize, graph: &KnowledgeGraph, llm: &L) -> Result<Self> {
        let mut pool = Self::new();
        let generator = PersonaGenerator::new();
        let mut generated_personas: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        // Get all entities from graph to use as persona anchors
        let entities = graph.get_all_entities();
        if entities.is_empty() {
            return Err(TeriError::Agent(
                "No entities available in graph for persona generation".to_string(),
            ));
        }

        // Generate N unique personas
        for i in 0..n {
            let mut attempts = 0;
            let max_attempts = 5; // Prevent infinite loops

            loop {
                // Cycle through entities if we need more personas than available entities
                let entity = &entities[i % entities.len()];

                let persona = generator.generate(graph, entity, llm).await.map_err(|e| {
                    TeriError::Agent(format!(
                        "Failed to generate persona for entity {}: {}",
                        entity.name, e
                    ))
                })?;

                // Create a unique identifier for the persona (name + role combination)
                let persona_id = format!("{}|{}", persona.name, persona.role);

                // Check if this persona is unique
                if !generated_personas.contains(&persona_id) {
                    generated_personas.insert(persona_id);
                    let agent = Agent::new(persona);
                    pool.add_agent(agent);
                    break;
                }

                attempts += 1;
                if attempts >= max_attempts {
                    // If we can't generate a unique persona after several attempts,
                    // create a variation by adding a suffix
                    let mut varied_persona = persona.clone();
                    varied_persona.name = format!("{} {}", varied_persona.name, attempts);
                    let varied_id = format!("{}|{}", varied_persona.name, varied_persona.role);
                    generated_personas.insert(varied_id);
                    let agent = Agent::new(varied_persona);
                    pool.add_agent(agent);
                    break;
                }
            }
        }

        Ok(pool)
    }

    /// Add a memory entry to the shared group memory
    pub async fn add_group_memory(&self, entry: MemoryEntry) {
        let mut group_memory = self.group_memory.write().await;

        // Check capacity BEFORE pushing to prevent temporary unbounded growth
        if group_memory.len() >= 1000 {
            let len = group_memory.len();
            group_memory.drain(0..len - 999); // Keep space for the new entry
        }

        group_memory.push(entry);
    }

    /// Get recent group memory entries
    pub async fn get_group_memory(&self, limit: usize) -> Vec<MemoryEntry> {
        let group_memory = self.group_memory.read().await;
        group_memory.iter().rev().take(limit).cloned().collect()
    }
}

/// Generates personas based on entities from the knowledge graph
pub struct PersonaGenerator {
    template: String,
}

impl PersonaGenerator {
    /// Create a new PersonaGenerator with the default embedded template
    pub fn new() -> Self {
        let template = include_str!("../../templates/persona_gen.jinja").to_string();
        Self { template }
    }

    /// Create a new PersonaGenerator with a custom template from file
    /// Falls back to embedded template if file loading fails
    pub fn from_file<P: AsRef<std::path::Path>>(template_path: P) -> Self {
        match std::fs::read_to_string(template_path) {
            Ok(template) => Self { template },
            Err(e) => {
                eprintln!(
                    "Warning: Failed to load template from file ({}), falling back to embedded template",
                    e
                );
                Self::new()
            }
        }
    }

    /// Create a new PersonaGenerator with a custom template string
    pub fn with_template(template: String) -> Self {
        Self { template }
    }

    /// Sanitize entity names to prevent template injection
    fn sanitize_entity_name(&self, name: &str) -> String {
        // Replace template-like patterns that could interfere with string replacement
        name.replace("{{", "")
            .replace("}}", "")
            .replace("{%", "")
            .replace("%}", "")
            // Also replace any newlines that could break template formatting
            .replace(['\n', '\r'], " ")
            // Trim multiple spaces
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Generate a persona based on an entity from the knowledge graph
    pub async fn generate<L: LlmClient>(
        &self,
        graph: &KnowledgeGraph,
        entity: &Entity,
        llm: &L,
    ) -> Result<Persona> {
        // Create a simple description based on entity connections
        let entity_description = self.generate_entity_description(graph, entity)?;

        // Sanitize entity name to prevent template injection
        let sanitized_name = self.sanitize_entity_name(&entity.name);

        // Render the template
        let prompt = self
            .template
            .replace("{{ entity_name }}", &sanitized_name)
            .replace("{{ entity_kind }}", &entity.kind.to_string())
            .replace("{{ entity_description }}", &entity_description);

        // Generate persona using LLM
        let response = llm.complete(&prompt).await?;

        // Parse the JSON response
        let persona: Persona = serde_json::from_str(&response)
            .map_err(|e| TeriError::Agent(format!("Failed to parse persona JSON: {}", e)))?;

        // Validate persona
        self.validate_persona(&persona)?;

        Ok(persona)
    }

    /// Generate a simple description of an entity based on its connections
    fn generate_entity_description(
        &self,
        graph: &KnowledgeGraph,
        entity: &Entity,
    ) -> Result<String> {
        let neighbors = graph.get_neighbors(entity.id).map_err(|e| {
            TeriError::Agent(format!("Failed to get neighbors for {}: {}", entity.name, e))
        })?;

        if neighbors.is_empty() {
            Ok(format!("{} is a {} with no known connections.", entity.name, entity.kind))
        } else {
            let neighbor_names: Vec<String> = neighbors
                .iter()
                .take(3) // Limit to avoid overly long descriptions
                .map(|n| n.name.clone())
                .collect();

            Ok(format!(
                "{} is a {} connected to: {}.",
                entity.name,
                entity.kind,
                neighbor_names.join(", ")
            ))
        }
    }

    /// Validate that a persona meets minimum requirements
    fn validate_persona(&self, persona: &Persona) -> Result<()> {
        if persona.name.trim().is_empty() {
            return Err(TeriError::Agent("Persona name cannot be empty".to_string()));
        }

        if persona.background.trim().is_empty() {
            return Err(TeriError::Agent("Persona background cannot be empty".to_string()));
        }

        if persona.traits.is_empty() || persona.traits.len() > 10 {
            return Err(TeriError::Agent("Persona must have between 1 and 10 traits".to_string()));
        }

        if persona.role.trim().is_empty() {
            return Err(TeriError::Agent("Persona role cannot be empty".to_string()));
        }

        Ok(())
    }
}

impl Default for PersonaGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Generates action prompts based on agent context and world state
pub struct ActionGenerator {
    template: String,
}

impl ActionGenerator {
    /// Create a new ActionGenerator with the default embedded template
    pub fn new() -> Self {
        let template = include_str!("../../templates/agent_action.jinja").to_string();
        Self { template }
    }

    /// Create a new ActionGenerator with a custom template from file
    /// Falls back to embedded template if file loading fails
    pub fn from_file<P: AsRef<std::path::Path>>(template_path: P) -> Self {
        match std::fs::read_to_string(template_path) {
            Ok(template) => Self { template },
            Err(e) => {
                eprintln!(
                    "Warning: Failed to load action template from file ({}), falling back to embedded template",
                    e
                );
                Self::new()
            }
        }
    }

    /// Generate a prompt for action generation based on agent and context
    pub fn generate_prompt(&self, agent: &Agent, context: &str) -> Result<String> {
        let env = Environment::new();

        // Parse recent events from context
        let recent_events = self.parse_recent_events(context);
        let relevant_memories = self.parse_relevant_memories(context);
        let world_variables = self.parse_world_variables(context);
        let world_tick = self.parse_world_tick(context);

        let template_context = context! {
            agent_name => &agent.persona.name,
            agent_role => &agent.persona.role,
            agent_state => format!("{:?}", agent.state),
            agent_background => &agent.persona.background,
            agent_traits => &agent.persona.traits,
            world_tick => world_tick,
            recent_events => recent_events,
            relevant_memories => relevant_memories,
            world_variables => world_variables,
        };

        let prompt = env
            .template_from_str(&self.template)
            .map_err(|e| TeriError::Agent(format!("Template parsing error: {}", e)))?
            .render(template_context)
            .map_err(|e| TeriError::Agent(format!("Template rendering error: {}", e)))?;

        Ok(prompt)
    }

    /// Parse recent events from context string
    fn parse_recent_events(&self, context: &str) -> Vec<String> {
        let mut events = Vec::new();
        if let Some(events_start) = context.find("Recent Events:")
            && let Some(events_end) = context[events_start..].find("\n\n")
        {
            let events_section = &context[events_start + 14..events_start + events_end];
            for line in events_section.lines() {
                if let Some(content) = line.strip_prefix("- ") {
                    events.push(content.to_string());
                }
            }
        }
        events
    }

    /// Parse relevant memories from context string
    fn parse_relevant_memories(&self, context: &str) -> Vec<MemoryEntry> {
        let mut memories = Vec::new();
        if let Some(memories_start) = context.find("Relevant Memories:")
            && let Some(memories_end) = context[memories_start..].find("\n\n")
        {
            let memories_section = &context[memories_start + 19..memories_start + memories_end];
            for line in memories_section.lines() {
                if let Some(content) = line.strip_prefix("- ") {
                    memories.push(MemoryEntry {
                        timestamp: Utc::now(),
                        content: content.to_string(),
                        importance: 0.7,
                    });
                }
            }
        }
        memories
    }

    /// Parse world variables from context string
    fn parse_world_variables(&self, context: &str) -> std::collections::HashMap<String, f32> {
        let mut variables = std::collections::HashMap::new();
        if let Some(vars_start) = context.find("World State:")
            && let Some(vars_end) = context[vars_start..].find("\n\n")
        {
            let vars_section = &context[vars_start + 12..vars_start + vars_end];
            for line in vars_section.lines() {
                if let Some(line_content) = line.strip_prefix("- ")
                    && let Some(colon_pos) = line_content.find(':')
                {
                    let key = line_content[..colon_pos].trim().to_string();
                    let value_str = line_content[colon_pos + 1..].trim();
                    if let Ok(value) = value_str.parse::<f32>() {
                        variables.insert(key, value);
                    }
                }
            }
        }
        variables
    }

    /// Parse world tick from context string
    fn parse_world_tick(&self, context: &str) -> u32 {
        if let Some(tick_start) = context.find("World Tick: ")
            && let Some(tick_end) = context[tick_start + 12..].find('\n')
        {
            let tick_str = &context[tick_start + 12..tick_start + 12 + tick_end];
            return tick_str.parse().unwrap_or(0);
        }
        0
    }
}

impl Default for ActionGenerator {
    fn default() -> Self {
        Self::new()
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
    use crate::graph::{EntityKind, KnowledgeGraph};
    use async_trait::async_trait;
    use std::pin::Pin;

    // Mock LLM for testing
    struct MockPersonaLlm {
        response: String,
    }

    impl MockPersonaLlm {
        fn new(response: &str) -> Self {
            Self { response: response.to_string() }
        }
    }

    #[async_trait]
    impl LlmClient for MockPersonaLlm {
        async fn complete(&self, _prompt: &str) -> Result<String> {
            Ok(self.response.clone())
        }

        async fn complete_json<T: serde::de::DeserializeOwned>(&self, _prompt: &str) -> Result<T> {
            Err(TeriError::Llm("Not implemented in mock".to_string()))
        }

        async fn stream(
            &self,
            _prompt: &str,
        ) -> Result<Pin<Box<dyn futures::Stream<Item = Result<String>> + Send>>> {
            Err(TeriError::Llm("Streaming not implemented in mock".to_string()))
        }
    }

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

    #[tokio::test]
    async fn test_persona_generator_creation() {
        let generator = PersonaGenerator::new();
        assert!(!generator.template.is_empty());
        assert!(generator.template.contains("persona generation system"));
    }

    #[tokio::test]
    async fn test_persona_generator_with_mock_llm() {
        let mock_response = r#"{
            "name": "Sarah Chen",
            "background": "An experienced project manager who has worked at Acme for 8 years.",
            "traits": ["organized", "detail-oriented", "collaborative"],
            "role": "Senior Project Manager"
        }"#;

        let mock_llm = MockPersonaLlm::new(mock_response);
        let generator = PersonaGenerator::new();

        // Create a test graph with an entity
        let mut graph = KnowledgeGraph::new();
        let entity = Entity {
            id: uuid::Uuid::new_v4(),
            name: "Acme Corporation".to_string(),
            kind: EntityKind::Organization,
        };
        graph.add_entity(entity.clone()).expect("Failed to add entity");

        let persona = generator
            .generate(&graph, &entity, &mock_llm)
            .await
            .expect("Failed to generate persona");

        assert_eq!(persona.name, "Sarah Chen");
        assert_eq!(persona.role, "Senior Project Manager");
        assert_eq!(persona.traits.len(), 3);
        assert!(persona.traits.contains(&"organized".to_string()));
    }

    #[tokio::test]
    async fn test_persona_generator_validation() {
        let generator = PersonaGenerator::new();

        // Test empty name
        let invalid_persona = Persona {
            name: "".to_string(),
            background: "Valid background".to_string(),
            traits: vec!["valid".to_string()],
            role: "Valid role".to_string(),
        };
        assert!(generator.validate_persona(&invalid_persona).is_err());

        // Test empty background
        let invalid_persona = Persona {
            name: "Valid Name".to_string(),
            background: "".to_string(),
            traits: vec!["valid".to_string()],
            role: "Valid role".to_string(),
        };
        assert!(generator.validate_persona(&invalid_persona).is_err());

        // Test too many traits
        let invalid_persona = Persona {
            name: "Valid Name".to_string(),
            background: "Valid background".to_string(),
            traits: (0..11).map(|i| format!("trait_{}", i)).collect(), // 11 traits
            role: "Valid role".to_string(),
        };
        assert!(generator.validate_persona(&invalid_persona).is_err());

        // Test valid persona
        let valid_persona = Persona {
            name: "Valid Name".to_string(),
            background: "Valid background".to_string(),
            traits: vec!["trait1".to_string(), "trait2".to_string()],
            role: "Valid role".to_string(),
        };
        assert!(generator.validate_persona(&valid_persona).is_ok());
    }

    #[tokio::test]
    async fn test_agent_pool_spawn_with_mock_llm() {
        let mock_response = r#"{
            "name": "Test Agent",
            "background": "A test agent for unit testing.",
            "traits": ["test-oriented", "methodical"],
            "role": "Test Subject"
        }"#;

        let mock_llm = MockPersonaLlm::new(mock_response);

        // Create a test graph with entities
        let mut graph = KnowledgeGraph::new();
        let entity1 = Entity {
            id: uuid::Uuid::new_v4(),
            name: "Entity1".to_string(),
            kind: EntityKind::Person,
        };
        let entity2 = Entity {
            id: uuid::Uuid::new_v4(),
            name: "Entity2".to_string(),
            kind: EntityKind::Organization,
        };
        graph.add_entity(entity1).expect("Failed to add entity1");
        graph.add_entity(entity2).expect("Failed to add entity2");

        // Spawn 2 agents
        let pool = AgentPool::spawn(2, &graph, &mock_llm).await.expect("Failed to spawn agents");

        assert_eq!(pool.len(), 2);

        // Verify agents have unique IDs
        let agents: Vec<_> = pool.iter().collect();
        assert_ne!(agents[0].id, agents[1].id);

        // Verify all agents have valid personas
        for agent in agents {
            assert!(!agent.persona.name.is_empty());
            assert!(!agent.persona.background.is_empty());
            assert!(!agent.persona.traits.is_empty());
            assert!(!agent.persona.role.is_empty());
        }
    }

    #[tokio::test]
    async fn test_agent_pool_group_memory() {
        let pool = AgentPool::new();

        // Add some group memories
        let memory1 = MemoryEntry {
            timestamp: chrono::Utc::now(),
            content: "Group memory 1".to_string(),
            importance: 0.8,
        };
        let memory2 = MemoryEntry {
            timestamp: chrono::Utc::now(),
            content: "Group memory 2".to_string(),
            importance: 0.9,
        };

        pool.add_group_memory(memory1.clone()).await;
        pool.add_group_memory(memory2.clone()).await;

        // Retrieve recent memories
        let recent = pool.get_group_memory(2).await;
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].content, "Group memory 2"); // Most recent first
        assert_eq!(recent[1].content, "Group memory 1");

        // Test limit
        let limited = pool.get_group_memory(1).await;
        assert_eq!(limited.len(), 1);
        assert_eq!(limited[0].content, "Group memory 2");
    }

    #[tokio::test]
    async fn test_agent_pool_spawn_empty_graph() {
        let mock_llm = MockPersonaLlm::new("{}");
        let empty_graph = KnowledgeGraph::new();

        let result = AgentPool::spawn(1, &empty_graph, &mock_llm).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No entities available"));
    }

    #[test]
    fn test_entity_description_generation() {
        let generator = PersonaGenerator::new();
        let mut graph = KnowledgeGraph::new();

        // Create an entity with no connections
        let isolated_entity = Entity {
            id: uuid::Uuid::new_v4(),
            name: "Isolated".to_string(),
            kind: EntityKind::Person,
        };
        graph
            .add_entity(isolated_entity.clone())
            .expect("Failed to add isolated entity");

        let description = generator
            .generate_entity_description(&graph, &isolated_entity)
            .expect("Failed to generate description");
        assert!(description.contains("no known connections"));

        // Create connected entities
        let connected_entity = Entity {
            id: uuid::Uuid::new_v4(),
            name: "Connected".to_string(),
            kind: EntityKind::Person,
        };
        let neighbor = Entity {
            id: uuid::Uuid::new_v4(),
            name: "Neighbor".to_string(),
            kind: EntityKind::Organization,
        };

        let connected_idx = graph
            .add_entity(connected_entity.clone())
            .expect("Failed to add connected entity");
        let neighbor_idx = graph.add_entity(neighbor.clone()).expect("Failed to add neighbor");

        graph.add_relation(
            connected_idx,
            neighbor_idx,
            crate::graph::Relation::new(crate::graph::RelationKind::RelatedTo, 0.8)
                .expect("Valid relation"),
        );

        let description = generator
            .generate_entity_description(&graph, &connected_entity)
            .expect("Failed to generate description");
        assert!(description.contains("connected to"));
        assert!(description.contains("Neighbor"));
    }

    #[test]
    fn test_template_sanitization() {
        let generator = PersonaGenerator::new();

        // Test entity names with template-like syntax
        let malicious_name = "Test {{ malicious }} {% injection %} \n\r\t";
        let sanitized = generator.sanitize_entity_name(malicious_name);

        // Should remove template syntax and whitespace
        assert!(!sanitized.contains("{{"));
        assert!(!sanitized.contains("}}"));
        assert!(!sanitized.contains("{%"));
        assert!(!sanitized.contains("%}"));
        assert!(!sanitized.contains('\n'));
        assert!(!sanitized.contains('\r'));
        assert!(!sanitized.contains('\t'));

        // Should preserve the actual content
        assert!(sanitized.contains("Test"));
        assert!(sanitized.contains("malicious"));
        assert!(sanitized.contains("injection"));
    }

    #[tokio::test]
    async fn test_persona_deduplication() {
        let mock_response = r#"{
            "name": "Duplicate Agent",
            "background": "An agent that would be duplicated.",
            "traits": ["duplicate", "test"],
            "role": "Test Subject"
        }"#;

        let mock_llm = MockPersonaLlm::new(mock_response);

        // Create a test graph with a single entity
        let mut graph = KnowledgeGraph::new();
        let entity = Entity {
            id: uuid::Uuid::new_v4(),
            name: "SingleEntity".to_string(),
            kind: EntityKind::Person,
        };
        graph.add_entity(entity).expect("Failed to add entity");

        // Spawn 3 agents - should create variations to avoid duplicates
        let pool = AgentPool::spawn(3, &graph, &mock_llm).await.expect("Failed to spawn agents");

        assert_eq!(pool.len(), 3);

        // Verify agents have unique personas
        let agents: Vec<_> = pool.iter().collect();
        let mut persona_names: Vec<String> =
            agents.iter().map(|a| a.persona.name.clone()).collect();

        // Sort and count unique names
        persona_names.sort();
        let unique_count = persona_names
            .iter()
            .zip(persona_names.iter().skip(1))
            .filter(|(a, b)| a != b)
            .count()
            + 1;

        // Should have at least 2 unique names (original + variations)
        assert!(unique_count >= 2);

        // Verify at least one agent has the original name
        assert!(persona_names.iter().any(|name| name.contains("Duplicate Agent")));

        // Verify at least one agent has a varied name (with numeric suffix)
        assert!(persona_names.iter().any(|name| name.chars().any(|c| c.is_ascii_digit())));
    }

    #[test]
    fn test_persona_generator_from_file() {
        // Test with non-existent file (should fall back to embedded template)
        let generator = PersonaGenerator::from_file("non_existent_template.jinja");
        assert!(!generator.template.is_empty());
        assert!(generator.template.contains("persona generation system"));
    }

    #[test]
    fn test_persona_generator_with_custom_template() {
        let custom_template =
            "Custom template for {{ entity_name }} ({{ entity_kind }})".to_string();
        let generator = PersonaGenerator::with_template(custom_template.clone());
        assert_eq!(generator.template, custom_template);
    }
}
