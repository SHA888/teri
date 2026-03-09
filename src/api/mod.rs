use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSimRequest {
    pub seed_path: String,
    pub query: String,
    pub agent_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSimResponse {
    pub sim_id: Uuid,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimStatusResponse {
    pub sim_id: Uuid,
    pub tick: u32,
    pub status: String,
    pub agent_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectRequest {
    pub variable: String,
    pub value: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub message: String,
    pub agent_id: Option<Uuid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub message: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SimStatus {
    Running,
    Completed,
    Failed,
}

pub struct ApiState {
    pub config: crate::Config,
}

impl ApiState {
    pub fn new(config: crate::Config) -> Self {
        Self { config }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_sim_request() {
        let req = CreateSimRequest {
            seed_path: "/path/to/seed.txt".to_string(),
            query: "What will happen?".to_string(),
            agent_count: Some(100),
        };

        assert_eq!(req.seed_path, "/path/to/seed.txt");
    }

    #[test]
    fn test_chat_request() {
        let req = ChatRequest {
            message: "Hello".to_string(),
            agent_id: Some(Uuid::new_v4()),
        };

        assert_eq!(req.message, "Hello");
    }
}
