use crate::error::Result;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEvent {
    pub tick: u32,
    pub description: String,
    pub significance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentHighlight {
    pub agent_id: Uuid,
    pub agent_name: String,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionReport {
    pub id: Uuid,
    pub summary: String,
    pub timeline: Vec<TimelineEvent>,
    pub agent_highlights: Vec<AgentHighlight>,
    pub confidence: f32,
    pub raw_query: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

pub struct ReportAgent;

impl ReportAgent {
    pub fn new() -> Self {
        Self
    }

    pub fn create_empty_report(query: String) -> PredictionReport {
        PredictionReport {
            id: Uuid::new_v4(),
            summary: String::new(),
            timeline: Vec::new(),
            agent_highlights: Vec::new(),
            confidence: 0.0,
            raw_query: query,
            created_at: chrono::Utc::now(),
        }
    }
}

impl Default for ReportAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeline_event_creation() {
        let event = TimelineEvent {
            tick: 5,
            description: "Something happened".to_string(),
            significance: 0.8,
        };

        assert_eq!(event.tick, 5);
        assert_eq!(event.significance, 0.8);
    }

    #[test]
    fn test_agent_highlight_creation() {
        let highlight = AgentHighlight {
            agent_id: Uuid::new_v4(),
            agent_name: "Alice".to_string(),
            summary: "Alice was very active".to_string(),
        };

        assert_eq!(highlight.agent_name, "Alice");
    }

    #[test]
    fn test_prediction_report_creation() {
        let report = ReportAgent::create_empty_report("What will happen?".to_string());
        assert_eq!(report.raw_query, "What will happen?");
        assert!(report.summary.is_empty());
    }
}
