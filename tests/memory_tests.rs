use futures::future::join_all;
use tempfile::TempDir;
use teri::memory::{MemoryEntry, MemoryStore};
use teri::sim::WorldSnapshot;
use uuid::Uuid;

#[tokio::test]
async fn test_snapshot_persistence() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().join("test_db");
    let store = MemoryStore::new(&db_path).expect("Failed to create memory store");
    let sim_id = Uuid::new_v4();
    let snapshot = WorldSnapshot {
        tick: 1,
        agents: std::collections::HashMap::new(),
        events: Vec::new(),
        variables: std::collections::HashMap::new(),
    };
    store.write_snapshot(sim_id, 1, &snapshot).await.expect("Write snapshot failed");
    let read = store.read_snapshot(sim_id, 1).await.expect("Read snapshot failed");
    assert_eq!(read.tick, snapshot.tick);
}

#[tokio::test]
async fn test_concurrent_access() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().join("test_db");
    let store = MemoryStore::new(&db_path).expect("Failed to create memory store");
    let agent_id = Uuid::new_v4();
    let mut futures = Vec::new();
    for i in 0..10 {
        let store_clone = store.clone();
        let entry = MemoryEntry {
            timestamp: chrono::Utc::now(),
            content: format!("Entry {}", i),
            importance: i as f32 * 0.1,
        };
        futures.push(tokio::spawn(async move {
            store_clone.write_ltm(agent_id, &entry).await.unwrap();
        }));
    }
    join_all(futures).await;
    let entries = store.read_ltm(agent_id, 20).await.expect("Read failed");
    assert_eq!(entries.len(), 10);
}

#[tokio::test]
async fn test_error_handling_invalid_path() {
    let invalid_path = if cfg!(windows) { "C:\\invalid_path\\?*" } else { "/root/invalid_path" };
    let result = MemoryStore::new(invalid_path);
    assert!(result.is_err());
}
