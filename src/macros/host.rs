use std::collections::HashMap;
use std::io::{BufRead, BufReader, PipeReader, PipeWriter, Write};
use std::sync::atomic::Ordering;
use std::sync::{Mutex, PoisonError};

use base_db::{Crate, all_crates};
use hir_def::nameres::crate_def_map;
use hir_def::{AssocItemId, CallableDefId, FunctionId};
use hir_expand::proc_macro::ProcMacroKind;
use hir_ty::db::HirDatabase;
use hir_ty::next_solver::{CallableIdWrapper, DbInterner, GenericArgs, Ty, TyKind};
use proc_macro_api::legacy_protocol::msg::{
    ExpandMacro, ExpandMacroData, ExpandMacroExtended, ExpnGlobals, FlatTree, Message, PanicMessage, Request,
    Response, deserialize_span_data_index_map, serialize_span_data_index_map,
};
use proc_macro_api::version::CURRENT_API_VERSION;
use span::Span;
use vfs::FileId;

use super::{Declared, declared};

struct Child {
    pid: i32,
    requests: PipeWriter,
    replies: BufReader<PipeReader>,
}

static ZYGOTE: Mutex<Option<Child>> = Mutex::new(None);

/// Replaces the zygote with a fork of this process; the fork handlers hold the engine lock across the fork.
pub fn refresh() {
    let mut zygote = ZYGOTE.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(old) = zygote.take() {
        end(old);
    }
    match fork_child(|requests, replies| run_zygote(requests, replies)) {
        Ok(child) => *zygote = Some(child),
        Err(reason) => eprintln!("proc macro zygote: {reason}"),
    }
}

/// Sends one request for the proc-macro crate at `root` through the zygote to that crate's host.
pub(super) fn ask(root: FileId, request: Request) -> Result<Response, String> {
    let mut zygote = ZYGOTE.lock().unwrap_or_else(PoisonError::into_inner);
    let zygote = zygote.as_mut().ok_or("the engine has not started a proc macro zygote")?;
    writeln!(zygote.requests, "{}", root.index()).map_err(|e| e.to_string())?;
    exchange(zygote, request)
}

fn exchange(child: &mut Child, request: Request) -> Result<Response, String> {
    request.write(&mut child.requests).map_err(|e| e.to_string())?;
    let mut buffer = String::new();
    Response::read(&mut child.replies, &mut buffer).map_err(|e| e.to_string())?.ok_or_else(|| "no reply".to_owned())
}

fn end(child: Child) -> i32 {
    let Child { pid, requests, replies } = child;
    drop((requests, replies));
    let mut status = 0;
    // SAFETY: pid is a child this process forked, and status is a live local.
    unsafe { libc::waitpid(pid, &mut status, 0) };
    status
}

fn fork_child(body: impl FnOnce(PipeReader, PipeWriter)) -> Result<Child, String> {
    let (request_read, request_write) = std::io::pipe().map_err(|e| e.to_string())?;
    let (reply_read, reply_write) = std::io::pipe().map_err(|e| e.to_string())?;
    // SAFETY: the forked child runs only this thread, uses state that the fork copied, and ends with _exit.
    let pid = unsafe { libc::fork() };
    if pid < 0 {
        return Err(format!("fork failed: {}", std::io::Error::last_os_error()));
    }
    if pid == 0 {
        drop((request_write, reply_read));
        body(request_read, reply_write);
        // SAFETY: _exit ends the forked child without running destructors that belong to the parent's state.
        unsafe { libc::_exit(0) }
    }
    drop((request_read, reply_write));
    Ok(Child { pid, requests: request_write, replies: BufReader::new(reply_read) })
}

fn run_zygote(requests: PipeReader, mut replies: PipeWriter) {
    let mut requests = BufReader::new(requests);
    let mut hosts = HashMap::new();
    let (mut header, mut buffer) = (String::new(), String::new());
    loop {
        header.clear();
        let Some(root) = requests.read_line(&mut header).ok().filter(|read| *read > 0).and_then(|_| header.trim_end().parse().ok()) else {
            return;
        };
        let Ok(Some(request)) = Request::read(&mut requests, &mut buffer) else {
            return;
        };
        let reply = relay(&mut hosts, FileId::from_raw(root), request);
        let reply = reply.unwrap_or_else(|reason| Response::ExpandMacroExtended(Err(PanicMessage(reason))));
        if reply.write(&mut replies).is_err() {
            return;
        }
    }
}

fn relay(hosts: &mut HashMap<FileId, Child>, root: FileId, request: Request) -> Result<Response, String> {
    if !hosts.contains_key(&root) {
        let host = fork_child(move |requests, replies| serve(root, requests, replies))?;
        hosts.insert(root, host);
    }
    let host = hosts.get_mut(&root).ok_or("no proc macro host")?;
    let answer = exchange(host, request);
    if let Err(reason) = &answer {
        let status = hosts.remove(&root).map_or(0, end);
        return Err(format!("the proc macro host exited with status {status:#x} ({reason})"));
    }
    answer
}

fn serve(root: FileId, requests: PipeReader, mut replies: PipeWriter) {
    crate::fault::install();
    let mut requests = BufReader::new(requests);
    let mut served = match Served::new(root) {
        Ok(served) => served,
        Err(reason) => return eprintln!("proc macro host: {reason}"),
    };
    let mut buffer = String::new();
    while let Ok(Some(Request::ExpandMacro(task))) = Request::read(&mut requests, &mut buffer) {
        let reply = Response::ExpandMacroExtended(served.expand(*task));
        if reply.write(&mut replies).is_err() {
            return;
        }
    }
}

struct Served {
    krate: Crate,
    declared: Vec<Declared>,
    clients: HashMap<String, usize>,
}

impl Served {
    fn new(root: FileId) -> Result<Self, String> {
        let found = crate::patch::with_engine(|db, _| {
            let krate = all_crates(db).iter().copied().find(|krate| krate.data(db).root_file_id == root)?;
            Some((krate, declared(db, krate)))
        });
        let (krate, declared) = found.ok_or("no crate has this root file")?;
        Ok(Served { krate, declared, clients: HashMap::new() })
    }

    fn expand(&mut self, task: ExpandMacro) -> Result<ExpandMacroExtended, PanicMessage> {
        let ExpandMacroData { macro_body, macro_name, attributes, has_global_spans, span_data_table } = task.data;
        let ExpnGlobals { def_site, call_site, mixed_site, .. } = has_global_spans;
        let mut table = deserialize_span_data_index_map(&span_data_table);
        let sites = [table[call_site], table[def_site], table[mixed_site]];
        let join = |first: Span, second: Span| first.join(second, |_, _| None).unwrap_or(second);
        let body = macro_body.to_tokenstream_resolved(CURRENT_API_VERSION, &table, join);
        let attribute = attributes.map(|it| it.to_tokenstream_resolved(CURRENT_API_VERSION, &table, join));
        let (address, kind) = self.client(&macro_name).map_err(PanicMessage)?;
        enter_env(&task.env, task.current_dir.as_deref());
        // SAFETY: address is the run function that expand1 or expand2 built from the sysroot's proc_macro, the source of this binary's proc_macro.
        let output = unsafe { proc_macro_srv::run_client(address, server_kind(kind), body, attribute, sites) };
        let output = output.map_err(|message| PanicMessage(message.into_string().unwrap_or_default()))?;
        let tree = FlatTree::from_tokenstream(output, CURRENT_API_VERSION, sites[0], &mut table);
        Ok(ExpandMacroExtended { tree, span_data_table: serialize_span_data_index_map(&table) })
    }

    fn client(&mut self, name: &str) -> Result<(usize, ProcMacroKind), String> {
        let declared = self.declared.iter().find(|it| it.name == name).ok_or_else(|| format!("no proc macro {name}"))?;
        let kind = declared.kind;
        if let Some(address) = self.clients.get(name) {
            return Ok((*address, kind));
        }
        let krate = self.krate;
        let code = crate::patch::with_engine(|db, backend| {
            let function = declared.function;
            let builder = client_builder(db, krate, kind)?;
            let interner = DbInterner::new_with(db, krate);
            let item = Ty::new(interner, TyKind::FnDef(CallableIdWrapper(CallableDefId::FunctionId(function)), GenericArgs::empty()));
            let (slot, _) = backend.instance(db, builder, GenericArgs::new_from_slice(&[item.into()]), function)?;
            Ok::<_, String>(backend.table()[slot].load(Ordering::Acquire))
        })?;
        let mut client = [0_u64; 4];
        crate::backend::enter(code, client.as_mut_ptr(), &0);
        self.clients.insert(name.to_owned(), client[0] as usize);
        Ok((client[0] as usize, kind))
    }
}

fn server_kind(kind: ProcMacroKind) -> proc_macro_srv::ProcMacroKind {
    match kind {
        ProcMacroKind::CustomDerive => proc_macro_srv::ProcMacroKind::CustomDerive,
        ProcMacroKind::Bang => proc_macro_srv::ProcMacroKind::Bang,
        ProcMacroKind::Attr => proc_macro_srv::ProcMacroKind::Attr,
    }
}

fn enter_env(env: &[(String, String)], current_dir: Option<&str>) {
    for (key, value) in env {
        // SAFETY: the forked host runs one thread, so nothing reads the environment while it changes.
        unsafe { std::env::set_var(key, value) };
    }
    if let Some(dir) = current_dir {
        let _ = std::env::set_current_dir(dir);
    }
}

fn client_builder(db: &dyn HirDatabase, krate: Crate, kind: ProcMacroKind) -> Result<FunctionId, String> {
    let wanted = if kind == ProcMacroKind::Attr { "expand2" } else { "expand1" };
    let proc_macro = krate.data(db).dependencies.iter().find(|dep| dep.name.symbol().as_str() == "proc_macro").ok_or("no proc_macro dependency")?;
    let map = crate_def_map(db, proc_macro.crate_id);
    let impls = map.modules().flat_map(|(_, module)| module.scope.impls());
    let found = impls.flat_map(|imp| imp.impl_items(db).items.iter().cloned().collect::<Vec<_>>()).find_map(|(name, item)| match item {
        AssocItemId::FunctionId(func) if name.as_str() == wanted => Some(func),
        _ => None,
    });
    found.ok_or_else(|| format!("no Client::{wanted} in proc_macro"))
}
