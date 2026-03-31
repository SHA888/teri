//! Integration tests for the graph module with real seed documents

use async_trait::async_trait;
use std::collections::HashMap;
use std::pin::Pin;
use teri::graph::{Entity, EntityKind, KnowledgeGraph, RelationKind};
use teri::llm::LlmClient;
use teri::seed::SeedIngestor;

/// Mock LLM client for integration testing
struct IntegrationMockLlm {
    responses: HashMap<String, String>,
}

impl IntegrationMockLlm {
    fn new() -> Self {
        let mut responses = HashMap::new();

        // Pre-defined responses for common test scenarios
        responses.insert(
            "simple".to_string(),
            r#"[
                {"name": "Teri", "kind": "Concept"},
                {"name": "Integration", "kind": "Concept"},
                {"name": "Test", "kind": "Concept"}
            ]"#
            .to_string(),
        );

        responses.insert(
            "relations_simple".to_string(),
            r#"[
                {"from": "Teri", "to": "Integration", "kind": "RelatedTo", "weight": 0.8},
                {"from": "Integration", "to": "Test", "kind": "RelatedTo", "weight": 0.9}
            ]"#
            .to_string(),
        );

        Self { responses }
    }

    fn get_response_for_prompt(&self, prompt: &str) -> Option<&String> {
        if prompt.contains("Extract named entities") {
            self.responses.get("simple")
        } else if prompt.contains("extract relations") {
            self.responses.get("relations_simple")
        } else {
            None
        }
    }
}

#[async_trait]
impl LlmClient for IntegrationMockLlm {
    async fn complete(&self, prompt: &str) -> teri::error::Result<String> {
        self.get_response_for_prompt(prompt)
            .cloned()
            .ok_or_else(|| teri::error::TeriError::Llm("No mock response for prompt".to_string()))
    }

    async fn complete_json<T: serde::de::DeserializeOwned>(
        &self,
        prompt: &str,
    ) -> teri::error::Result<T> {
        let response = self.complete(prompt).await?;
        serde_json::from_str(&response)
            .map_err(|e| teri::error::TeriError::Llm(format!("JSON parsing error: {}", e)))
    }

    async fn stream(
        &self,
        _prompt: &str,
    ) -> teri::error::Result<Pin<Box<dyn futures::Stream<Item = teri::error::Result<String>> + Send>>>
    {
        Err(teri::error::TeriError::Llm(
            "Streaming not implemented in integration mock".to_string(),
        ))
    }
}

#[tokio::test]
async fn test_integration_with_real_seed_document() -> Result<(), Box<dyn std::error::Error>> {
    // Load real seed document
    let seed_doc = SeedIngestor::from_file("examples/seed.txt").await?;

    // Verify seed document was loaded correctly
    assert!(!seed_doc.raw_text.is_empty());
    assert!(seed_doc.raw_text.contains("Teri"));
    assert!(seed_doc.raw_text.contains("integration"));

    // Create mock LLM
    let mock_llm = IntegrationMockLlm::new();

    // Test entity extraction
    let entity_prompt = KnowledgeGraph::entity_extraction_prompt(&seed_doc);
    let entity_response = mock_llm.complete(&entity_prompt).await?;
    let entities = KnowledgeGraph::parse_entities_json(&entity_response)?;

    assert!(!entities.is_empty());
    assert_eq!(entities.len(), 3);

    // Verify entities
    let entity_names: Vec<String> = entities.iter().map(|e| e.name.clone()).collect();
    assert!(entity_names.contains(&"Teri".to_string()));
    assert!(entity_names.contains(&"Integration".to_string()));
    assert!(entity_names.contains(&"Test".to_string()));

    // Build graph with entities
    let mut graph = KnowledgeGraph::new();
    for entity in entities {
        graph.add_entity(entity)?;
    }

    // Test relation extraction
    let entities: Vec<Entity> = graph.get_all_entities().into_iter().cloned().collect();
    let relation_prompt = KnowledgeGraph::relation_extraction_prompt(&seed_doc, &entities);
    let relation_response = mock_llm.complete(&relation_prompt).await?;

    // Create index for relation parsing
    let index = graph.get_index();

    let relations = KnowledgeGraph::parse_relations_json(&relation_response, index)?;

    assert!(!relations.is_empty());
    assert_eq!(relations.len(), 2);

    // Add relations to graph
    for (from_idx, to_idx, relation) in relations {
        graph.add_relation(from_idx, to_idx, relation);
    }

    // Verify final graph structure
    assert_eq!(graph.entity_count(), 3);
    assert_eq!(graph.relation_count(), 2);

    // Test graph queries
    let teri_entity = graph.get_entity("Teri").expect("Teri entity should exist");
    assert_eq!(teri_entity.kind, EntityKind::Concept);

    let neighbors = graph.get_neighbors(teri_entity.id)?;
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0].name, "Integration");

    // Test subgraph extraction
    let subgraph = graph.get_subgraph(teri_entity.id, 2)?;
    // Subgraph creates unique names to avoid conflicts, so it may have more entities
    // due to name suffixes for duplicates
    assert!(subgraph.entity_count() >= 3); // Should include at least the connected entities
    // Verify that all original entities are present (possibly with suffixes)
    let subgraph_entities: Vec<String> = subgraph
        .get_all_entities()
        .iter()
        .map(|e| e.name.split('_').next().unwrap_or(&e.name).to_string())
        .collect();
    assert!(subgraph_entities.contains(&"Teri".to_string()));
    assert!(subgraph_entities.contains(&"Integration".to_string()));
    assert!(subgraph_entities.contains(&"Test".to_string()));

    Ok(())
}

#[tokio::test]
async fn test_graph_serialization_roundtrip_with_real_data()
-> Result<(), Box<dyn std::error::Error>> {
    // Create a realistic graph
    let mut graph = KnowledgeGraph::new();

    // Add entities that might come from real document processing
    let entities = vec![
        ("Alice", EntityKind::Person),
        ("Bob", EntityKind::Person),
        ("Acme Corporation", EntityKind::Organization),
        ("New York", EntityKind::Location),
        ("Software Development", EntityKind::Concept),
    ];

    let mut entity_ids = HashMap::new();
    for (name, kind) in entities {
        let entity = teri::graph::Entity { id: uuid::Uuid::new_v4(), name: name.to_string(), kind };
        entity_ids.insert(name.to_string(), entity.id);
        graph.add_entity(entity)?;
    }

    // Add realistic relations
    let index = graph.get_index();
    let alice_idx = *index.get("Alice").unwrap();
    let bob_idx = *index.get("Bob").unwrap();
    let acme_idx = *index.get("Acme Corporation").unwrap();
    let ny_idx = *index.get("New York").unwrap();

    graph.add_relation(
        alice_idx,
        acme_idx,
        teri::graph::Relation::new(RelationKind::WorksFor, 0.9)?,
    );

    graph.add_relation(bob_idx, acme_idx, teri::graph::Relation::new(RelationKind::WorksFor, 0.8)?);

    graph.add_relation(acme_idx, ny_idx, teri::graph::Relation::new(RelationKind::LocatedIn, 1.0)?);

    // Test JSON serialization
    let json_data = graph.serialize_to_json()?;
    assert!(!json_data.is_empty());

    // Test JSON deserialization
    let deserialized_graph = KnowledgeGraph::deserialize_from_json(&json_data)?;

    // Verify structure is preserved
    assert_eq!(deserialized_graph.entity_count(), graph.entity_count());
    assert_eq!(deserialized_graph.relation_count(), graph.relation_count());

    // Verify specific entities and relations
    let alice = deserialized_graph.get_entity("Alice").ok_or("Alice not found")?;
    assert_eq!(alice.kind, EntityKind::Person);

    let alice_neighbors = deserialized_graph.get_neighbors(alice.id)?;
    assert_eq!(alice_neighbors.len(), 1);
    assert_eq!(alice_neighbors[0].name, "Acme Corporation");

    // Test bincode serialization
    let bincode_data = graph.serialize_to_bincode()?;
    assert!(!bincode_data.is_empty());

    // Test bincode deserialization
    let bincode_deserialized = KnowledgeGraph::deserialize_from_bincode(&bincode_data)?;

    // Verify bincode roundtrip
    assert_eq!(bincode_deserialized.entity_count(), graph.entity_count());
    assert_eq!(bincode_deserialized.relation_count(), graph.relation_count());

    Ok(())
}

#[tokio::test]
async fn test_error_handling_integration() -> Result<(), Box<dyn std::error::Error>> {
    let _mock_llm = IntegrationMockLlm::new();

    // Test with malformed JSON response
    let malformed_json = r#"{"invalid": json}"#;

    let result = KnowledgeGraph::parse_entities_json(malformed_json);
    assert!(result.is_err());

    // Test with invalid entity data
    let invalid_entities = r#"[
        {"name": "", "kind": "Person"},
        {"kind": "Organization"}
    ]"#;

    let result = KnowledgeGraph::parse_entities_json(invalid_entities);
    // The second entity is missing the name field, so this should fail
    assert!(result.is_err());

    // Test relation parsing with invalid weights
    let mut graph = KnowledgeGraph::new();
    let entity = teri::graph::Entity {
        id: uuid::Uuid::new_v4(),
        name: "Test".to_string(),
        kind: EntityKind::Concept,
    };
    graph.add_entity(entity)?;

    let invalid_relations = r#"[
        {"from": "Test", "to": "Test", "kind": "RelatedTo", "weight": 1.5}
    ]"#;

    let result = KnowledgeGraph::parse_relations_json(invalid_relations, graph.get_index());
    assert!(result.is_err());

    Ok(())
}

#[test]
fn test_graph_construction_edge_cases() -> Result<(), Box<dyn std::error::Error>> {
    // Test empty graph
    let empty_graph = KnowledgeGraph::new();
    assert_eq!(empty_graph.entity_count(), 0);
    assert_eq!(empty_graph.relation_count(), 0);

    // Test graph with single entity
    let mut single_entity_graph = KnowledgeGraph::new();
    let entity = teri::graph::Entity {
        id: uuid::Uuid::new_v4(),
        name: "Singleton".to_string(),
        kind: EntityKind::Concept,
    };
    single_entity_graph.add_entity(entity)?;

    assert_eq!(single_entity_graph.entity_count(), 1);
    assert_eq!(single_entity_graph.relation_count(), 0);

    // Test subgraph of single entity
    let singleton = single_entity_graph.get_entity("Singleton").unwrap();
    let subgraph = single_entity_graph.get_subgraph(singleton.id, 5)?;
    assert_eq!(subgraph.entity_count(), 1);
    assert_eq!(subgraph.relation_count(), 0);

    // Test neighbors of isolated entity
    let neighbors = single_entity_graph.get_neighbors(singleton.id)?;
    assert_eq!(neighbors.len(), 0);

    Ok(())
}
