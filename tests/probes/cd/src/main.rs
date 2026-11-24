use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "demo", version = "1.0")]
struct Cli {
    #[arg(short, long, default_value_t = 4)]
    jobs: usize,
    #[arg(short, long)]
    release: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    Build {
        #[arg(default_value = ".")]
        path: String,
    },
    Test {
        #[arg(short, long)]
        filter: Option<String>,
    },
}

fn main() {
    let a = Cli::parse_from(["demo", "-j", "8", "--release", "build", "app"]);
    let b = Cli::parse_from(["demo", "test", "--filter", "math"]);
    let a_path = match &a.cmd {
        Cmd::Build { path } => path.as_str(),
        Cmd::Test { .. } => "",
    };
    let b_filter = match &b.cmd {
        Cmd::Test { filter } => filter.as_deref().unwrap_or(""),
        Cmd::Build { .. } => "",
    };
    println!("{} {} {} {} {}", a.jobs, a.release, a_path, b.jobs, b_filter);
}
