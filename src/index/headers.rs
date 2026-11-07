use std::collections::{HashMap, HashSet};

use base_db::relevant_crates;
use hir_def::nameres::crate_def_map;
use hir_def::signatures::{EnumSignature, ImplSignature, StructSignature, TraitSignature, UnionSignature};
use hir_def::{AdtId, ImplId, Lookup};
use hir_ty::attach_db;
use hir_ty::db::HirDatabase;
use hir_ty::next_solver::{Ty, TyKind};
use rustc_type_ir::inherent::{AdtDef as _, IntoKind};
use vfs::{FileId, Vfs};

use super::{Head, Header, Kind, Scanned, scan_all};

const SHOWN: usize = 40;
const PRIMITIVES: [&str; 20] = [
    "bool", "char", "str", "i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64", "u128", "usize",
    "f16", "f32", "f64", "f128", "Self",
];

#[derive(Default)]
struct Table {
    files: HashMap<FileId, HashSet<Header>>,
    listed: HashMap<Header, usize>,
    expansions: HashMap<Head, HashSet<String>>,
}

impl Table {
    fn load(db: &dyn HirDatabase, vfs: &Vfs) -> Self {
        let mut scanned = Vec::new();
        for (file, path) in vfs.iter() {
            if !path.name_and_extension().is_some_and(|(_, extension)| extension == Some("rs")) {
                continue;
            }
            if relevant_crates(db, file).is_empty() {
                continue;
            }
            scanned.push((file, scan_all(db.file_text(file).text(db))));
        }
        Self::build(scanned)
    }

    fn build(scanned: Vec<(FileId, Scanned)>) -> Self {
        let mut declared: HashSet<String> = PRIMITIVES.iter().map(|name| (*name).to_owned()).collect();
        for (_, found) in &scanned {
            declared.extend(found.aliases.iter().map(|alias| alias.name.clone()));
            let types = found.entries.iter().filter(|entry| declares_type(entry.kind));
            declared.extend(types.map(|entry| entry.name.clone()));
        }
        let untrusted = untrusted_names(&scanned);
        let mut table = Self::default();
        for (file, found) in scanned {
            for mut header in found.headers {
                let unsure = |name: &String| !declared.contains(name) || untrusted.contains(name);
                if matches!(&header.head, Head::Name(name) if unsure(name)) {
                    header.head = Head::Unknown;
                }
                *table.listed.entry(header.clone()).or_default() += 1;
                table.files.entry(file).or_default().insert(header);
            }
            for alias in found.aliases {
                table.expansions.entry(alias.head).or_default().insert(alias.name);
            }
        }
        table
    }

    fn closure(&self, head: Head) -> HashSet<Head> {
        let mut seen = HashSet::new();
        let mut queue = vec![head];
        while let Some(next) = queue.pop() {
            if let Some(names) = self.expansions.get(&next) {
                queue.extend(names.iter().cloned().map(Head::Name).filter(|name| !seen.contains(name)));
            }
            seen.insert(next);
        }
        seen
    }

    fn keys(&self, wanted: &Header) -> Vec<Header> {
        let traits = self.closure(Head::Name(wanted.trait_name.clone()));
        let mut heads = self.closure(wanted.head.clone());
        heads.insert(Head::Unknown);
        let mut keys = Vec::new();
        for known in &traits {
            let Head::Name(trait_name) = known else { continue };
            keys.extend(heads.iter().map(|head| Header { trait_name: trait_name.clone(), head: head.clone() }));
        }
        keys
    }
}

fn untrusted_names(scanned: &[(FileId, Scanned)]) -> HashSet<String> {
    let macros = || scanned.iter().flat_map(|(_, found)| &found.macros);
    let mut writers: HashSet<&str> = macros().filter(|facts| facts.declares_alias).map(|facts| facts.name.as_str()).collect();
    loop {
        let before = writers.len();
        for facts in macros() {
            if facts.calls.iter().any(|call| writers.contains(call.name.as_str())) {
                writers.insert(facts.name.as_str());
            }
        }
        if writers.len() == before {
            break;
        }
    }
    let inner = macros().flat_map(|facts| &facts.calls);
    let uses = scanned.iter().flat_map(|(_, found)| &found.uses).chain(inner);
    uses.filter(|call| writers.contains(call.name.as_str())).flat_map(|call| call.args.iter().cloned()).collect()
}

fn declares_type(kind: Kind) -> bool {
    matches!(kind, Kind::Struct | Kind::Enum | Kind::Union | Kind::TypeAlias)
}

fn adt_name(db: &dyn HirDatabase, adt: AdtId) -> String {
    match adt {
        AdtId::StructId(id) => StructSignature::of(db, id).name.as_str().to_owned(),
        AdtId::EnumId(id) => EnumSignature::of(db, id).name.as_str().to_owned(),
        AdtId::UnionId(id) => UnionSignature::of(db, id).name.as_str().to_owned(),
    }
}

fn head_of<'db>(db: &'db dyn HirDatabase, ty: Ty<'db>) -> Head {
    match ty.kind() {
        TyKind::Adt(adt, _) => Head::Name(adt_name(db, adt.def_id())),
        TyKind::Bool => Head::Name("bool".to_owned()),
        TyKind::Char => Head::Name("char".to_owned()),
        TyKind::Str => Head::Name("str".to_owned()),
        TyKind::Int(int) => Head::Name(int.name_str().to_owned()),
        TyKind::Uint(int) => Head::Name(int.name_str().to_owned()),
        TyKind::Float(float) => Head::Name(float.name_str().to_owned()),
        TyKind::Ref(..) => Head::Ref,
        TyKind::RawPtr(..) => Head::Ptr,
        TyKind::Slice(_) => Head::Slice,
        TyKind::Array(..) => Head::Array,
        TyKind::Tuple(_) => Head::Tuple,
        TyKind::FnPtr(..) => Head::Fn,
        TyKind::Dynamic(..) => Head::Dyn,
        TyKind::Never => Head::Never,
        TyKind::Param(_) => Head::Param,
        TyKind::Foreign(_) => Head::Unknown,
        _ => Head::Unknown,
    }
}

fn resolved(db: &dyn HirDatabase, id: ImplId) -> Option<Header> {
    let self_ty = db.impl_self_ty(id).skip_binder();
    if matches!(self_ty.kind(), TyKind::Error(_)) {
        return None;
    }
    let trait_name = match db.impl_trait(id) {
        Some(found) => TraitSignature::of(db, found.skip_binder().def_id.0).name.as_str().to_owned(),
        None if ImplSignature::of(db, id).target_trait.is_some() => return None,
        None => String::new(),
    };
    Some(Header { trait_name, head: head_of(db, self_ty) })
}

#[derive(Default)]
struct Tally {
    source: usize,
    from_macro: usize,
    unresolved: usize,
    misses: usize,
    real_by_key: HashMap<Header, usize>,
}

fn check(db: &dyn HirDatabase, vfs: &Vfs, table: &Table, tally: &mut Tally, id: ImplId) {
    let file = id.lookup(db).id.file_id;
    let Some(source) = file.file_id() else {
        tally.from_macro += 1;
        return;
    };
    tally.source += 1;
    let file = source.file_id(db);
    let Some(wanted) = resolved(db, id) else {
        tally.unresolved += 1;
        return;
    };
    *tally.real_by_key.entry(wanted.clone()).or_default() += 1;
    let headers = table.files.get(&file);
    if !table.keys(&wanted).iter().any(|key| headers.is_some_and(|found| found.contains(key))) {
        if tally.misses < SHOWN {
            println!("header_miss {} {wanted:?} scanned={headers:?}", vfs.file_path(file));
        }
        tally.misses += 1;
    }
}

pub fn run(db: &dyn HirDatabase, vfs: &Vfs) {
    attach_db(db, || {
        let table = Table::load(db, vfs);
        let mut tally = Tally::default();
        for krate in base_db::all_crates(db).iter().copied() {
            for (_, module) in crate_def_map(db, krate).modules() {
                for id in module.scope.impls() {
                    check(db, vfs, &table, &mut tally, id);
                }
            }
        }
        let listed: usize = tally
            .real_by_key
            .keys()
            .map(|wanted| table.keys(wanted).iter().filter_map(|key| table.listed.get(key)).sum::<usize>())
            .sum();
        let is_unknown = |entry: &(&Header, &usize)| entry.0.head == Head::Unknown;
        let unknown: usize = table.listed.iter().filter(is_unknown).map(|entry| *entry.1).sum();
        println!(
            "headers source={} misses={} unresolved={} from_macro={} keys={} listed={listed} unknown_heads={unknown} aliases={}",
            tally.source,
            tally.misses,
            tally.unresolved,
            tally.from_macro,
            tally.real_by_key.len(),
            table.expansions.values().map(HashSet::len).sum::<usize>(),
        );
    });
}
