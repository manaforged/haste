use std::any::Any;
use std::sync::Arc;

use base_db::{Crate, Env, SourceDatabase};
use hir_def::nameres::crate_def_map;
use hir_def::{FunctionId, Lookup, MacroId};
use hir_expand::proc_macro::{CrateProcMacros, ProcMacro, ProcMacroExpander, ProcMacroExpansionError, ProcMacroKind, ProcMacros};
use ide_db::RootDatabase;
use base_db::salsa::{Durability, Setter};
use intern::Symbol;
use paths::{AbsPath, AbsPathBuf};
use vfs::Vfs;
use proc_macro_api::legacy_protocol::msg::{
    ExpandMacro, ExpandMacroData, ExpnGlobals, FlatTree, PanicMessage, Request, Response, SpanDataIndexMap,
    deserialize_span_data_index_map, serialize_span_data_index_map,
};
use proc_macro_api::version::CURRENT_API_VERSION;
use span::Span;
use vfs::FileId;

pub mod census;
mod host;

pub use host::refresh;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Declared {
    name: String,
    kind: ProcMacroKind,
    function: FunctionId,
}

#[derive(Debug, PartialEq, Eq)]
struct Expander {
    root: FileId,
    path: AbsPathBuf,
    name: String,
}

/// Gives every registered proc-macro crate the macros its name resolution declares, each run by a forked host.
pub fn declare(db: &mut RootDatabase, vfs: &Vfs) {
    let mut by_crate = ProcMacros::get(db).by_crate(db).clone();
    for (krate, macros) in by_crate.iter_mut() {
        let root = krate.data(db).root_file_id;
        let Some(path) = vfs.file_path(root).as_path().map(AbsPath::to_path_buf) else {
            continue;
        };
        let made = declared(db, *krate).into_iter().map(|declared| ProcMacro {
            name: Symbol::intern(&declared.name),
            kind: declared.kind,
            expander: Arc::new(Expander { root, path: path.clone(), name: declared.name }),
            disabled: false,
        });
        *macros = triomphe::Arc::new(CrateProcMacros::new(Ok(made.collect())));
    }
    ProcMacros::get(db).set_by_crate(db).with_durability(Durability::HIGH).to(by_crate);
}

fn declared(db: &dyn SourceDatabase, krate: Crate) -> Vec<Declared> {
    let map = crate_def_map(db, krate);
    let found = map[map.root].scope.macros().filter_map(|(name, item)| match item.def {
        MacroId::ProcMacroId(id) => {
            let function = map.proc_macro_as_fn(id)?;
            Some(Declared { name: name.as_str().to_owned(), kind: id.lookup(db).kind, function })
        }
        _ => None,
    });
    found.collect()
}

impl Expander {
    fn request(&self, subtree: &tt::TopSubtree, attrs: Option<&tt::TopSubtree>, env: &Env, sites: [Span; 3], current_dir: String) -> Request {
        let mut table = SpanDataIndexMap::default();
        let [def_site, call_site, mixed_site] = sites.map(|site| table.insert_full(site).0);
        let data = ExpandMacroData {
            macro_body: FlatTree::from_subtree(subtree.view(), CURRENT_API_VERSION, &mut table),
            macro_name: self.name.clone(),
            attributes: attrs.map(|attrs| FlatTree::from_subtree(attrs.view(), CURRENT_API_VERSION, &mut table)),
            has_global_spans: ExpnGlobals { serialize: true, def_site, call_site, mixed_site },
            span_data_table: serialize_span_data_index_map(&table),
        };
        let lib = self.path.clone().into();
        Request::ExpandMacro(Box::new(ExpandMacro { lib, env: env.clone().into(), current_dir: Some(current_dir), data }))
    }
}

impl ProcMacroExpander for Expander {
    fn expand(
        &self,
        _db: &dyn SourceDatabase,
        subtree: &tt::TopSubtree,
        attrs: Option<&tt::TopSubtree>,
        env: &Env,
        def_site: Span,
        call_site: Span,
        mixed_site: Span,
        current_dir: String,
    ) -> Result<tt::TopSubtree, ProcMacroExpansionError> {
        let request = self.request(subtree, attrs, env, [def_site, call_site, mixed_site], current_dir);
        match host::ask(self.root, request).map_err(ProcMacroExpansionError::System)? {
            Response::ExpandMacroExtended(Ok(done)) => {
                let table = deserialize_span_data_index_map(&done.span_data_table);
                Ok(FlatTree::to_subtree_resolved(done.tree, CURRENT_API_VERSION, &table))
            }
            Response::ExpandMacroExtended(Err(PanicMessage(message))) => Err(ProcMacroExpansionError::Panic(message)),
            _ => Err(ProcMacroExpansionError::System("the proc macro host gave an unexpected reply".to_owned())),
        }
    }

    fn eq_dyn(&self, other: &dyn ProcMacroExpander) -> bool {
        (other as &dyn Any).downcast_ref::<Self>() == Some(self)
    }
}
