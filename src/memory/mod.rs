use crate::error::{Result, TeriError};
use crate::sim::WorldSnapshot;
use redb::{Database, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

const AGENT_LTM_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("agent_ltm");
const WORLD_SNAPSHOT_TABLE: TableDefinition<&str, &[u8]> = TableDefinition::new("world_snapshots");

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub content: String,
    pub importance: f32,
}

pub struct MemoryStore {
    db: Arc<Database>,
}

impl MemoryStore {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let db = Database::create(path.as_ref())
            .map_err(|e| TeriError::Memory(format!("Failed to open redb: {e}")))?;

        let write_txn = db
            .begin_write()
            .map_err(|e| TeriError::Memory(format!("Failed to begin write transaction: {e}")))?;
        {
            let _ = write_txn
                .open_table(AGENT_LTM_TABLE)
                .map_err(|e| TeriError::Memory(format!("Failed to open agent_ltm table: {e}")))?;
            let _ = write_txn.open_table(WORLD_SNAPSHOT_TABLE).map_err(|e| {
                TeriError::Memory(format!("Failed to open world_snapshots table: {e}"))
            })?;
        }
        write_txn
            .commit()
            .map_err(|e| TeriError::Memory(format!("Failed to commit transaction: {e}")))?;

        Ok(Self { db: Arc::new(db) })
    }

    pub async fn write_ltm(&self, agent_id: Uuid, entry: &MemoryEntry) -> Result<()> {
        let ts = entry.timestamp.timestamp();
        let key = format!("agent:{agent_id}:ltm:{ts}");
        let value = serde_json::to_vec(entry)
            .map_err(|e| TeriError::Memory(format!("Serialization error: {e}")))?;

        let db = self.db.clone();
        tokio::task::spawn_blocking(move || {
            let write_txn = db.begin_write().map_err(|e| {
                TeriError::Memory(format!("Failed to begin write transaction: {e}"))
            })?;
            {
                let mut table = write_txn
                    .open_table(AGENT_LTM_TABLE)
                    .map_err(|e| TeriError::Memory(format!("Failed to open table: {e}")))?;
                table
                    .insert(key.as_str(), value.as_slice())
                    .map_err(|e| TeriError::Memory(format!("Write error: {e}")))?;
            }
            write_txn
                .commit()
                .map_err(|e| TeriError::Memory(format!("Failed to commit: {e}")))?;
            Ok(())
        })
        .await
        .map_err(|e| TeriError::Memory(format!("Task join error: {e}")))?
    }

    pub async fn read_ltm(&self, agent_id: Uuid, limit: usize) -> Result<Vec<MemoryEntry>> {
        let prefix = format!("agent:{agent_id}:ltm:");
        let db = self.db.clone();

        tokio::task::spawn_blocking(move || {
            let read_txn = db
                .begin_read()
                .map_err(|e| TeriError::Memory(format!("Failed to begin read transaction: {e}")))?;
            let table = read_txn
                .open_table(AGENT_LTM_TABLE)
                .map_err(|e| TeriError::Memory(format!("Failed to open table: {e}")))?;

            let mut entries = Vec::new();
            let mut iter = table
                .iter()
                .map_err(|e| TeriError::Memory(format!("Failed to create iterator: {e}")))?;

            while let Some(Ok((key, value))) = iter.next() {
                let key_str = key.value();
                if key_str.starts_with(&prefix) {
                    let entry: MemoryEntry = serde_json::from_slice(value.value())
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
            let write_txn = db.begin_write().map_err(|e| {
                TeriError::Memory(format!("Failed to begin write transaction: {e}"))
            })?;
            {
                let mut table = write_txn
                    .open_table(WORLD_SNAPSHOT_TABLE)
                    .map_err(|e| TeriError::Memory(format!("Failed to open table: {e}")))?;
                table
                    .insert(key.as_str(), value.as_slice())
                    .map_err(|e| TeriError::Memory(format!("Write error: {e}")))?;
            }
            write_txn
                .commit()
                .map_err(|e| TeriError::Memory(format!("Failed to commit: {e}")))?;
            Ok(())
        })
        .await
        .map_err(|e| TeriError::Memory(format!("Task join error: {e}")))?
    }

    pub async fn read_snapshot(&self, sim_id: Uuid, tick: u32) -> Result<WorldSnapshot> {
        let key = format!("world:{sim_id}:tick:{tick:010}");
        let db = self.db.clone();

        tokio::task::spawn_blocking(move || {
            let read_txn = db
                .begin_read()
                .map_err(|e| TeriError::Memory(format!("Failed to begin read transaction: {e}")))?;
            let table = read_txn
                .open_table(WORLD_SNAPSHOT_TABLE)
                .map_err(|e| TeriError::Memory(format!("Failed to open table: {e}")))?;

            let value = table
                .get(key.as_str())
                .map_err(|e| TeriError::Memory(format!("Read error: {e}")))?
                .ok_or_else(|| TeriError::Memory(format!("Snapshot not found: {key}")))?;

            bincode::deserialize(value.value())
                .map_err(|e| TeriError::Memory(format!("Deserialization error: {e}")))
        })
        .await
        .map_err(|e| TeriError::Memory(format!("Task join error: {e}")))?
    }

    pub async fn read_history(&self, sim_id: Uuid) -> Result<Vec<WorldSnapshot>> {
        let prefix = format!("world:{sim_id}:tick:");
        let db = self.db.clone();

        tokio::task::spawn_blocking(move || {
            let read_txn = db
                .begin_read()
                .map_err(|e| TeriError::Memory(format!("Failed to begin read transaction: {e}")))?;
            let table = read_txn
                .open_table(WORLD_SNAPSHOT_TABLE)
                .map_err(|e| TeriError::Memory(format!("Failed to open table: {e}")))?;

            let mut snapshots = Vec::new();
            let mut iter = table
                .iter()
                .map_err(|e| TeriError::Memory(format!("Failed to create iterator: {e}")))?;

            while let Some(Ok((key, value))) = iter.next() {
                let key_str = key.value();
                if key_str.starts_with(&prefix) {
                    let snapshot: WorldSnapshot = bincode::deserialize(value.value())
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
