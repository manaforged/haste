use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Result;

use super::{Outcome, Run, directives, rustc_text};

enum Verdict {
    Agree,
    Differ(Vec<String>),
    Failed,
    NoRecord,
}

fn real_records(project: &Path) -> Result<Vec<PathBuf>> {
    let cargo = Path::new(rustc_text(&["--print", "sysroot"])?.trim()).join("bin/cargo");
    let check = Command::new(cargo).args(["check", "--message-format=json", "--quiet", "--keep-going"]).current_dir(project).output()?;
    let lines = String::from_utf8_lossy(&check.stdout).into_owned();
    let messages = lines.lines().filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok());
    let executed = messages.filter(|message| message["reason"] == "build-script-executed");
    Ok(executed.filter_map(|message| message["out_dir"].as_str().map(PathBuf::from)).collect())
}

fn record_name(out_dir: &Path) -> Option<String> {
    let unit = out_dir.parent()?.file_name()?.to_str()?;
    Some(unit.rsplit_once('-')?.0.to_owned())
}

fn recorded(real_out: &Path, out: &Path) -> Option<BTreeSet<String>> {
    let text = std::fs::read_to_string(real_out.parent()?.join("output")).ok()?;
    Some(directives(&text.replace(&*real_out.to_string_lossy(), &out.to_string_lossy())))
}

fn read_tree(root: &Path, dir: &Path, into: &mut BTreeMap<PathBuf, Vec<u8>>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            read_tree(root, &path, into)?;
        } else {
            into.insert(path.strip_prefix(root)?.to_owned(), std::fs::read(&path)?);
        }
    }
    Ok(())
}

fn with_path(bytes: Vec<u8>, from: &str, to: &str) -> Vec<u8> {
    match String::from_utf8(bytes) {
        Ok(text) => text.replace(from, to).into_bytes(),
        Err(error) => error.into_bytes(),
    }
}

fn file_differences(real_out: &Path, out: &Path) -> Result<Vec<String>> {
    let (mut real, mut ours) = (BTreeMap::new(), BTreeMap::new());
    read_tree(real_out, real_out, &mut real)?;
    read_tree(out, out, &mut ours)?;
    let (from, to) = (real_out.to_string_lossy(), out.to_string_lossy());
    let real: BTreeMap<PathBuf, Vec<u8>> = real.into_iter().map(|(path, bytes)| (path, with_path(bytes, &from, &to))).collect();
    let names: BTreeSet<&PathBuf> = real.keys().chain(ours.keys()).collect();
    let differing = names.into_iter().filter(|name| real.get(*name) != ours.get(*name));
    Ok(differing.map(|name| format!("file {}", name.display())).collect())
}

fn compare(output: &str, outcome: &Outcome, records: &mut Vec<PathBuf>) -> Result<Verdict> {
    let ours = directives(output);
    let mut first_real = None;
    let mut first_files = None;
    let mut position = 0;
    while position < records.len() {
        let record = &records[position];
        if record_name(record).as_deref() == Some(&outcome.name) && let Some(real) = recorded(record, &outcome.out) {
            if real == ours {
                let files = file_differences(record, &outcome.out)?;
                if files.is_empty() {
                    records.remove(position);
                    return Ok(Verdict::Agree);
                }
                first_files.get_or_insert(files);
            }
            first_real.get_or_insert(real);
        }
        position += 1;
    }
    if let Some(files) = first_files {
        return Ok(Verdict::Differ(files));
    }
    let Some(real) = first_real else { return Ok(Verdict::NoRecord) };
    let extra = ours.difference(&real).map(|line| format!("+{line}"));
    Ok(Verdict::Differ(extra.chain(real.difference(&ours).map(|line| format!("-{line}"))).collect()))
}

#[derive(Default)]
struct Tally {
    agree: usize,
    differ: usize,
    failed: usize,
    no_record: usize,
}

impl Tally {
    fn add(&mut self, label: &str, verdict: Verdict) {
        match verdict {
            Verdict::Agree => self.agree += 1,
            Verdict::Failed => self.failed += 1,
            Verdict::NoRecord => {
                self.no_record += 1;
                println!("script_no_record {label}");
            }
            Verdict::Differ(lines) => {
                self.differ += 1;
                println!("script_differ {label}: {lines:?}");
            }
        }
    }
}

pub fn run(outcomes: &[Outcome], project: &Path) -> Result<()> {
    let mut records = real_records(project)?;
    let mut tally = Tally::default();
    for outcome in outcomes {
        let verdict = match &outcome.run {
            Run::Done(output) => compare(output, outcome, &mut records)?,
            Run::Failed(_) => Verdict::Failed,
        };
        tally.add(&outcome.label, verdict);
    }
    for record in &records {
        println!("script_unmatched {}", record.display());
    }
    println!(
        "scripts total={} agree={} differ={} failed={} no_record={} unmatched_records={}",
        outcomes.len(),
        tally.agree,
        tally.differ,
        tally.failed,
        tally.no_record,
        records.len()
    );
    Ok(())
}
