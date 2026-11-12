//! `haste debug code`: the machine code the backend makes for each function a binary's `main` reaches, with the definition site by which it joins a prep manifest's records.

use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Context as _, Result};
use haste_codegen::Target;

use crate::graph::{Targets, Workspace};
use crate::image::Image;
use crate::lower::SymbolSource;
use crate::sema::Context;
use crate::sema::reach::{Reached, calls};
use crate::sema::substs::Substs;

/// The file `write` makes in the directory it is given.
const FILE: &str = "code.tsv";
const HEADER: &str = "index\tmodule\tname\tdefinition\ttype_args\tinline_callees\tbytes\tcode\trelocations\n";
const NONE: &str = "-";
const DATA: &str = "data";
const RUNTIME: &str = "runtime";

/// One function the backend compiled on its own: its reached index, how many of its callees it may copy inline, its code, and each relocation as its offset and the reached index and name of the function it calls, or what else it names.
struct Compiled {
    index: usize,
    inline: usize,
    code: Vec<u8>,
    relocations: String,
}

/// Compiles each function the binary's `main` reaches under the current profile on its own, and writes its identity (module, name, definition site, type argument count) and its machine code to `code.tsv` in `dir`.
pub fn write(workspace: &Workspace, targets: &Targets, dir: &Path) -> Result<()> {
    let mut compilation = crate::macros::load_eager(workspace, targets, &[])?;
    let main = crate::build::main_item(workspace, &compilation.outline).context("no member binary has a `main`")?;
    let text = crate::macros::reach_resolved(&mut compilation, |_| vec![(main, Substs::EMPTY)], |mut cx, reached| table(&mut cx, &reached))?;
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join(FILE), text)?;
    std::mem::forget(compilation);
    Ok(())
}

fn table(cx: &mut Context<'_>, reached: &Reached) -> Result<String> {
    let image = Image::new(reached)?;
    let mut compiled = Vec::new();
    let mut failed = 0;
    for index in 0..reached.list.len() {
        let ((item, _), Ok(_)) = &reached.list[index] else { continue };
        if cx.foreign(*item) {
            continue;
        }
        match compile(cx, reached, &image, index) {
            Ok((code, relocations)) => compiled.push(Compiled { index, inline: inline_callees(cx, reached, index), code, relocations }),
            Err(error) => {
                failed += 1;
                eprintln!("code: {}: {error:#}", cx.name(*item));
            }
        }
    }
    let mut text = String::from(HEADER);
    for one in &compiled {
        row(cx, reached, one, &mut text)?;
    }
    eprintln!("code {} compiled, {failed} failed, {} reached", compiled.len(), reached.list.len());
    Ok(text)
}

fn compile(cx: &mut Context<'_>, reached: &Reached, image: &Image, index: usize) -> Result<(Vec<u8>, String)> {
    let out = crate::lower::lower_stencil(cx, reached, image, index)?;
    let code = haste_codegen::compile(out.func, Target::host())?;
    let mut relocations = String::new();
    for reloc in code.relocs() {
        match reloc.symbol_index().and_then(|at| out.relocs.get(at)) {
            Some(SymbolSource::Slot(callee)) => write!(relocations, "{}:{callee}/{} ", reloc.offset, reached.list.get(*callee).map_or(NONE, |((item, _), _)| cx.name(*item)))?,
            Some(_) => write!(relocations, "{}:{DATA} ", reloc.offset)?,
            None => write!(relocations, "{}:{RUNTIME} ", reloc.offset)?,
        }
    }
    Ok((code.bytes().to_vec(), relocations))
}

/// How many functions that reached function `index` calls lowering may copy inline into it.
fn inline_callees(cx: &mut Context<'_>, reached: &Reached, index: usize) -> usize {
    let callees: Vec<usize> = calls(cx, &reached.list[index].1).filter_map(|instance| reached.index.get(&instance).copied()).collect();
    callees.into_iter().filter(|&callee| crate::lower::may_inline(cx, reached, callee)).count()
}

fn row(cx: &mut Context<'_>, reached: &Reached, one: &Compiled, text: &mut String) -> Result<()> {
    let ((item, substs), _) = reached.list[one.index];
    let module = cx.module_of(item).map_or_else(|_| cx.outline.roots[item.krate as usize].name.clone(), |at| cx.outline.module_path(at));
    let args = cx.types.substs(substs).len();
    let definition = place(cx, item).unwrap_or_else(|| NONE.to_owned());
    write!(text, "{}\t{module}\t{}\t{definition}\t{args}\t{}\t{}\t", one.index, cx.name(item), one.inline, one.code.len())?;
    for byte in &one.code {
        write!(text, "{byte:02x}")?;
    }
    writeln!(text, "\t{}", one.relocations.trim_end())?;
    Ok(())
}

/// The file of item `item` and the byte offset of its name there, as `path:offset`, when the name is in the file's own text.
fn place(cx: &Context<'_>, item: crate::names::ItemRef) -> Option<String> {
    let (source, entry) = cx.item(item);
    let name = entry.name?;
    let span = source.span(name, name);
    let lo = (span.as_ptr() as usize).checked_sub(source.text.as_ptr() as usize)?;
    (item.outer().is_none() && source.base.is_none() && lo + span.len() <= source.text.len()).then(|| format!("{}:{lo}", source.path.display()))
}
