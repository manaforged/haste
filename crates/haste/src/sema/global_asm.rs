//! The module-level `global_asm!` blocks of the program: each block's template text and `sym` operands, found by the symbols it makes global.

use std::sync::Arc;

use anyhow::{Result, bail};
use rustc_hash::FxHashMap as HashMap;
use rustc_lexer::TokenKind;

use super::Context;
use super::infer::{Res, Shape, Typed};
use super::substs::Substs;
use crate::names::path::Path;
use crate::names::{Def, ItemRef, ModRef, plain};
use crate::outline::item::ItemKind;
use crate::outline::source::Source;
use crate::outline::tree::Waiting;

/// One `global_asm!` block: its template strings joined by newlines, and each `sym` operand's name with the function it names.
#[derive(Debug)]
pub struct Block {
    pub text: String,
    pub symbols: Vec<(String, ItemRef)>,
}

/// Every `global_asm!` block whose text and operands read, by each symbol it makes global, without the object format's prefix; and why each other block did not read.
#[derive(Debug, Default)]
pub struct Table {
    by_name: HashMap<String, Arc<Block>>,
    unread: Vec<String>,
}

impl Context<'_> {
    /// The `global_asm!` block that defines the symbol the foreign function `item` links to, when one does.
    pub fn global_asm(&mut self, item: ItemRef) -> Option<Arc<Block>> {
        if !self.foreign(item) || self.item(item).1.kind != ItemKind::Fn {
            return None;
        }
        let name = self.symbol_name(item)?;
        self.global_asm_table().by_name.get(&name).cloned()
    }

    /// Why the `global_asm!` blocks that did not read failed.
    pub fn unread_global_asm(&mut self) -> Vec<String> {
        self.global_asm_table().unread.clone()
    }

    fn global_asm_table(&mut self) -> Arc<Table> {
        if let Some(table) = &self.global_asm {
            return Arc::clone(table);
        }
        let before = self.asks.count();
        let table = Arc::new(table(self));
        if self.asks.count() == before {
            self.global_asm = Some(Arc::clone(&table));
        }
        table
    }
}

/// Gives the foreign function `item` the shape of the `global_asm!` block that defines its symbol, which calls each function the block's `sym` operands name.
pub(super) fn bind(cx: &mut Context<'_>, item: ItemRef, typed: &mut Typed) {
    if let Some(block) = cx.global_asm(item) {
        typed.shape = Shape::Asm(
            block
                .symbols
                .iter()
                .map(|&(_, callee)| Res::Fn(callee, Substs::EMPTY))
                .collect(),
        );
    }
}

fn table(cx: &mut Context<'_>) -> Table {
    let outline = cx.outline;
    let prefix = crate::host_target().symbol_prefix;
    let mut table = Table::default();
    for (krate, tree) in outline.crates.iter().enumerate() {
        for call in &tree.global_asm {
            let block = match block(cx, krate as u32, call) {
                Ok(block) => Arc::new(block),
                Err(error) => {
                    table.unread.push(format!(
                        "{}: global_asm!: {error:#}",
                        call.file.source.path.display()
                    ));
                    continue;
                }
            };
            for name in globals(&block.text).filter_map(|global| global.strip_prefix(prefix)) {
                table.by_name.insert(name.to_owned(), Arc::clone(&block));
            }
        }
    }
    table
}

fn globals(text: &str) -> impl Iterator<Item = &str> {
    text.lines().filter_map(|line| {
        let (directive, name) = line.trim().split_once(char::is_whitespace)?;
        matches!(directive, ".globl" | ".global").then(|| name.trim())
    })
}

fn block(cx: &mut Context<'_>, krate: u32, call: &Waiting) -> Result<Block> {
    let src = &call.file.source;
    let Some(open) = call.file.items.items[call.at as usize].body else {
        bail!("no arguments")
    };
    let home = ModRef {
        krate,
        module: call.owner as u32,
    };
    let args = src.args(open);
    let mut lines = Vec::new();
    for &range in &args {
        let Some(line) = cx.template_text(home, src, range)? else {
            break;
        };
        lines.push(line);
    }
    if lines.is_empty() {
        bail!("no template string");
    }
    let mut symbols = Vec::new();
    for &range in &args[lines.len()..] {
        let (name, path) = operand(src, range, symbols.len())?;
        symbols.push((name, sym_target(cx, home, src, path)?));
    }
    Ok(Block {
        text: lines.join("\n"),
        symbols,
    })
}

/// The name of the `sym` operand at `from..to`, its position `index` when it has none, and the tokens of its path.
fn operand(src: &Source, (from, to): (u32, u32), index: usize) -> Result<(String, (u32, u32))> {
    if src.text(from) == "sym" && to > from + 1 {
        return Ok((index.to_string(), (from + 1, to)));
    }
    if to > from + 3 && src.kind(from + 1) == TokenKind::Eq && src.text(from + 2) == "sym" {
        return Ok((plain(src.text(from)).to_owned(), (from + 3, to)));
    }
    bail!(
        "only `sym` operands are supported, not `{}`",
        spell(src, (from, to))
    )
}

/// The function that the path at `from..to` names in the value namespace of module `home`.
fn sym_target(
    cx: &Context<'_>,
    home: ModRef,
    src: &Source,
    (from, to): (u32, u32),
) -> Result<ItemRef> {
    let paths = cx.paths(home.krate);
    let words =
        (from..to).filter(|&at| matches!(src.kind(at), TokenKind::Ident | TokenKind::RawIdent));
    let segments: Vec<_> = words
        .map(|at| paths.symbols.intern(plain(src.text(at))))
        .collect();
    let dollar = if src.kind(from) == TokenKind::Dollar {
        src.dollar(from)
    } else {
        None
    };
    let path = Path {
        segments: &segments,
        absolute: src.kind(from) == TokenKind::Colon,
        dollar,
    };
    let found = match segments.split_last() {
        Some((last, [])) => paths.first(*last, home, &path).values,
        Some((last, prefix)) => paths
            .resolve(
                &Path {
                    segments: prefix,
                    ..path
                },
                home,
            )
            .and_then(|owner| paths.member(owner, *last).values),
        None => None,
    };
    match found.map(|binding| binding.def) {
        Some(Def::Item(item)) if cx.item(item).1.kind == ItemKind::Fn => Ok(item),
        _ => bail!("`sym {}` names no function", spell(src, (from, to))),
    }
}

fn spell(src: &Source, (from, to): (u32, u32)) -> String {
    (from..to).map(|at| src.text(at)).collect()
}
