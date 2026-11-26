use corelib::{Job, score};

pub fn run_batch(raw: &str) -> (i64, String) {
    let jobs: Vec<Job> = serde_json::from_str(raw).expect("parse jobs");
    let total: i64 = jobs.iter().map(score).sum();
    let names = jobs
        .iter()
        .map(|j| j.label.as_str())
        .collect::<Vec<_>>()
        .join("+");
    (total, names)
}
