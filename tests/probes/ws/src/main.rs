use corelib::Job;

fn main() {
    let batch = vec![
        Job {
            id: 2,
            weight: 10,
            label: "parse".to_owned(),
        },
        Job {
            id: 3,
            weight: 7,
            label: "emit".to_owned(),
        },
    ];
    let raw = serde_json::to_string(&batch).expect("encode batch");
    let (total, names) = worker::run_batch(&raw);
    println!("{total} {names}");
}
