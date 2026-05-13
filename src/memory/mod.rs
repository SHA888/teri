use crate::error::{Result, TeriError};
use crate::sim::WorldSnapshot;
use rocksdb::{DB as RocksDB, IteratorMode};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

// Key schema constants
// agent:{uuid}:ltm:{timestamp} → MemoryEntry
pub const AGENT_LTM_KEY_PREFIX: &str = "agent";
// world:{sim_id}:tick:{n} → WorldSnapshot
pub const WORLD_SNAPSHOT_KEY_PREFIX: &str = "world";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub content: String,
    pub importance: f32,
}

pub struct MemoryStore {
    // RocksDB instance for all memory operations
    db: Arc<RocksDB>,
}

impl MemoryStore {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        // Ensure the directory exists
        let rocks_path = path.as_ref().join("rocksdb");
        std::fs::create_dir_all(&rocks_path)
            .map_err(|e| TeriError::Memory(format!("Failed to create rocksdb dir: {e}")))?;
        let db = RocksDB::open_default(&rocks_path)
            .map_err(|e| TeriError::Memory(format!("Failed to open rocksdb: {e}")))?;
        Ok(Self { db: Arc::new(db) })
    }

    pub async fn write_ltm(&self, agent_id: Uuid, entry: &MemoryEntry) -> Result<()> {
        let ts = entry.timestamp.timestamp();
        let key = format!("agent:{agent_id}:ltm:{ts}");
        let value = serde_json::to_vec(entry)
            .map_err(|e| TeriError::Memory(format!("Serialization error: {e}")))?;
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            db.put(key.as_bytes(), &value)
                .map_err(|e| TeriError::Memory(format!("Write error: {e}")))
        })
        .await
        .map_err(|e| TeriError::Memory(format!("Task join error: {e}")))?
    }

    pub async fn read_ltm(&self, agent_id: Uuid, limit: usize) -> Result<Vec<MemoryEntry>> {
        let prefix = format!("agent:{agent_id}:ltm:");
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let mut entries = Vec::new();
            let iter = db.iterator(IteratorMode::Start);
            for item in iter {
                let (k, v) = item.map_err(|e| TeriError::Memory(format!("Iterator error: {e}")))?;
                let key_str = std::str::from_utf8(&k)
                    .map_err(|e| TeriError::Memory(format!("Invalid UTF8 key: {e}")))?;
                if key_str.starts_with(&prefix) {
                    let entry: MemoryEntry = serde_json::from_slice(&v)
                        .map_err(|e| TeriError::Memory(format!("Deserialization error: {e}")))?;
                    entries.push(entry);
                    if entries.len() >= limit {
                        break;
                    }
                }
            }
            Ok(entries)
        })
        .await
        .map_err(|e| TeriError::Memory(format!("Task join error: {e}")))?
    }

    // Stub for future full‑text query on long‑term memory
    pub async fn query_ltm(&self, _agent_id: Uuid, _query: &str) -> Result<Vec<MemoryEntry>> {
        // TODO: integrate a vector‑search or simple substring filter.
        // For now we return an empty vector to satisfy the API.
        Ok(Vec::new())
    }

    pub async fn write_snapshot(
        &self,
        sim_id: Uuid,
        tick: u32,
        snapshot: &WorldSnapshot,
    ) -> Result<()> {
        let key = format!("world:{sim_id}:tick:{tick:010}");
        let value = bincode::serialize(snapshot)
            .map_err(|e| TeriError::Memory(format!("Serialization error: {e}")))?;
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            db.put(key.as_bytes(), &value)
                .map_err(|e| TeriError::Memory(format!("Write error: {e}")))
        })
        .await
        .map_err(|e| TeriError::Memory(format!("Task join error: {e}")))?
    }

    pub async fn read_snapshot(&self, sim_id: Uuid, tick: u32) -> Result<WorldSnapshot> {
        let key = format!("world:{sim_id}:tick:{tick:010}");
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let v = db
                .get(key.as_bytes())
                .map_err(|e| TeriError::Memory(format!("Read error: {e}")))?
                .ok_or_else(|| TeriError::Memory(format!("Snapshot not found: {key}")))?;
            bincode::deserialize(&v)
                .map_err(|e| TeriError::Memory(format!("Deserialization error: {e}")))
        })
        .await
        .map_err(|e| TeriError::Memory(format!("Task join error: {e}")))?
    }

    pub async fn read_history(&self, sim_id: Uuid) -> Result<Vec<WorldSnapshot>> {
        let prefix = format!("world:{sim_id}:tick:");
        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let mut snapshots = Vec::new();
            let iter = db.iterator(IteratorMode::Start);
            for item in iter {
                let (k, v) = item.map_err(|e| TeriError::Memory(format!("Iterator error: {e}")))?;
                let key_str = std::str::from_utf8(&k)
                    .map_err(|e| TeriError::Memory(format!("Invalid UTF8 key: {e}")))?;
                if key_str.starts_with(&prefix) {
                    let snapshot: WorldSnapshot = bincode::deserialize(&v)
                        .map_err(|e| TeriError::Memory(format!("Deserialization error: {e}")))?;
                    snapshots.push(snapshot);
                }
            }
            Ok(snapshots)
        })
        .await
        .map_err(|e| TeriError::Memory(format!("Task join error: {e}")))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_memory_store_creation() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let db_path = temp_dir.path().join("test.redb");
        let _store = MemoryStore::new(&db_path).expect("Failed to create memory store");
    }

    #[tokio::test]
    async fn test_write_and_read_ltm() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let db_path = temp_dir.path().join("test.redb");
        let store = MemoryStore::new(&db_path).expect("Failed to create memory store");

        let agent_id = Uuid::new_v4();
        let entry = MemoryEntry {
            timestamp: chrono::Utc::now(),
            content: "Test memory".to_string(),
            importance: 0.8,
        };

        store.write_ltm(agent_id, &entry).await.expect("Failed to write LTM");

        let entries = store.read_ltm(agent_id, 10).await.expect("Failed to read LTM");

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].content, "Test memory");
    }
}
