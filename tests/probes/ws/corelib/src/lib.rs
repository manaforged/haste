use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Job {
    pub id: u32,
    pub weight: i64,
    pub label: String,
}

pub fn score(job: &Job) -> i64 {
    job.weight * i64::from(job.id) + job.label.len() as i64
}
