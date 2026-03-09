use crate::error::{Result, TeriError};
use petgraph::graph::{Graph, NodeIndex};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EntityKind {
    Person,
    Organization,
    Location,
    Concept,
    Event,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: Uuid,
    pub name: String,
    pub kind: EntityKind,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RelationKind {
    WorksFor,
    LocatedIn,
    RelatedTo,
    Causes,
    Affects,
    Other,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    pub kind: RelationKind,
    pub weight: f32,
}

pub struct KnowledgeGraph {
    inner: Graph<Entity, Relation>,
    index: HashMap<String, NodeIndex>,
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        Self {
            inner: Graph::new(),
            index: HashMap::new(),
        }
    }

    pub fn add_entity(&mut self, entity: Entity) -> NodeIndex {
        let node_idx = self.inner.add_node(entity.clone());
        self.index.insert(entity.name.clone(), node_idx);
        node_idx
    }

    pub fn add_relation(&mut self, from: NodeIndex, to: NodeIndex, relation: Relation) {
        self.inner.add_edge(from, to, relation);
    }

    pub fn get_entity(&self, name: &str) -> Option<&Entity> {
        self.index
            .get(name)
            .and_then(|idx| self.inner.node_weight(*idx))
    }

    pub fn get_neighbors(&self, entity_name: &str) -> Result<Vec<&Entity>> {
        let idx = self
            .index
            .get(entity_name)
            .ok_or_else(|| TeriError::Graph(format!("Entity not found: {}", entity_name)))?;

        let neighbors = self
            .inner
            .neighbors(*idx)
            .filter_map(|n| self.inner.node_weight(n))
            .collect();

        Ok(neighbors)
    }

    pub fn get_all_entities(&self) -> Vec<&Entity> {
        self.inner.node_weights().collect()
    }

    pub fn entity_count(&self) -> usize {
        self.inner.node_count()
    }

    pub fn relation_count(&self) -> usize {
        self.inner.edge_count()
    }
}

impl Default for KnowledgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_graph_creation() {
        let graph = KnowledgeGraph::new();
        assert_eq!(graph.entity_count(), 0);
        assert_eq!(graph.relation_count(), 0);
    }

    #[test]
    fn test_add_entity() {
        let mut graph = KnowledgeGraph::new();
        let entity = Entity {
            id: Uuid::new_v4(),
            name: "Alice".to_string(),
            kind: EntityKind::Person,
        };

        graph.add_entity(entity);
        assert_eq!(graph.entity_count(), 1);
    }

    #[test]
    fn test_get_entity() {
        let mut graph = KnowledgeGraph::new();
        let entity = Entity {
            id: Uuid::new_v4(),
            name: "Alice".to_string(),
            kind: EntityKind::Person,
        };

        graph.add_entity(entity.clone());
        let retrieved = graph.get_entity("Alice");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "Alice");
    }

    #[test]
    fn test_add_relation() {
        let mut graph = KnowledgeGraph::new();
        let alice = Entity {
            id: Uuid::new_v4(),
            name: "Alice".to_string(),
            kind: EntityKind::Person,
        };
        let bob = Entity {
            id: Uuid::new_v4(),
            name: "Bob".to_string(),
            kind: EntityKind::Person,
        };

        let alice_idx = graph.add_entity(alice);
        let bob_idx = graph.add_entity(bob);

        let relation = Relation {
            kind: RelationKind::RelatedTo,
            weight: 0.8,
        };

        graph.add_relation(alice_idx, bob_idx, relation);
        assert_eq!(graph.relation_count(), 1);
    }
}
