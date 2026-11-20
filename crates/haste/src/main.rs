#[cfg(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu"))]
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn main() -> anyhow::Result<()> {
    haste::main()
}
