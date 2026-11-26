use anyhow::{Context, Result};
use haste_codegen::{Arch, RelocKind, patch_reloc};
use rustc_hash::{FxHashMap, FxHashSet};

use super::plan::{Plan, resolve_sym};
use super::{Buf, CodeReloc, Executable, LOOP_PAGE, SectionAddrs, Symbol, shims, tls};

pub(crate) struct PatchCtx<'a> {
    pub(crate) text_base: usize,
    pub(crate) text_va: u64,
    pub(crate) pie_arm64: bool,
    pub(crate) addrs: SectionAddrs,
    pub(crate) base_va: u64,
    pub(crate) shim_offs: &'a [usize],
    pub(crate) plan: &'a Plan<'a>,
    pub(crate) stub_va: &'a dyn Fn(&str) -> Result<u64>,
    pub(crate) got_va: &'a dyn Fn(&str) -> Result<u64>,
    pub(crate) veneers: &'a Veneers,
}

#[derive(Default)]
pub(crate) struct Veneers {
    pub(crate) pre_len: usize,
    pub(crate) post_len: usize,
    pre: Vec<(Symbol, u64)>,
    post: Vec<(Symbol, u64)>,
    pre_vas: FxHashMap<Symbol, u64>,
    post_vas: FxHashMap<Symbol, u64>,
}

impl Veneers {
    fn resolve(&self, pc_va: u64, value: u64, sym: &Symbol) -> Option<u64> {
        if in_bl_range(pc_va, value) {
            return Some(value);
        }
        self.pre_vas
            .get(sym)
            .copied()
            .filter(|&va| in_bl_range(pc_va, va))
            .or_else(|| {
                self.post_vas
                    .get(sym)
                    .copied()
                    .filter(|&va| in_bl_range(pc_va, va))
            })
    }

    pub(crate) fn emit_pre(
        &self,
        resolve: &dyn Fn(&Symbol) -> Result<u64>,
        out: &mut Buf,
    ) -> Result<()> {
        emit_veneer_list(&self.pre, self.pre_len, resolve, out)
    }

    pub(crate) fn emit_post(
        &self,
        resolve: &dyn Fn(&Symbol) -> Result<u64>,
        out: &mut Buf,
    ) -> Result<()> {
        emit_veneer_list(&self.post, self.post_len, resolve, out)
    }
}

fn in_bl_range(pc_va: u64, target_va: u64) -> bool {
    let delta = target_va as i64 - pc_va as i64;
    delta & 3 == 0 && (-0x800_0000..0x800_0000).contains(&delta)
}

fn emit_veneer_list(
    list: &[(Symbol, u64)],
    total_len: usize,
    resolve: &dyn Fn(&Symbol) -> Result<u64>,
    out: &mut Buf,
) -> Result<()> {
    let start = out.pos();
    for (sym, va) in list {
        shims::emit_arm64_veneer(*va, resolve(sym)?, out);
    }
    out.0.resize(start + total_len, 0);
    Ok(())
}

pub(crate) fn plan_veneers(
    exe: &Executable,
    plan: &Plan<'_>,
    base_va: u64,
    head_off: usize,
    shim_offs: &[usize],
    calc_stubs_off: impl Fn(usize, usize) -> usize,
) -> Result<Veneers> {
    let lead =
        |veneers: usize| (head_off + veneers * 12).next_multiple_of(LOOP_PAGE as usize) - head_off;
    if exe.target.arch != Arch::Arm64
        || (head_off + exe.text.len() + plan.imports.len() * 12) < 0x700_0000
    {
        return Ok(Veneers {
            pre_len: lead(0),
            ..Veneers::default()
        });
    }
    let post_rel = exe.text.len().next_multiple_of(16);
    let mut pre_syms = Vec::new();
    let mut post_syms = Vec::new();
    let mut pre_seen = FxHashSet::default();
    let mut post_seen = FxHashSet::default();
    loop {
        let pre_len = lead(pre_syms.len());
        let post_len = (post_syms.len() * 12).next_multiple_of(16);
        let text_off = head_off + pre_len;
        let stubs_off = calc_stubs_off(pre_len, post_len);
        let addrs = SectionAddrs {
            text: base_va + text_off as u64,
            ..SectionAddrs::default()
        };
        let stub_va = |name: &str| Ok(base_va + (stubs_off + plan.import_idx(name)? * 12) as u64);
        let mut added = false;
        for reloc in &exe.relocs {
            if reloc.kind != RelocKind::Arm64Call26 {
                continue;
            }
            let pc_va = addrs.text + u64::from(reloc.offset);
            let target_va = resolve_sym(&reloc.symbol, addrs, base_va, shim_offs, plan, &stub_va)?;
            if in_bl_range(pc_va, target_va) {
                continue;
            }
            if reloc.offset as usize >= 0x700_0000 {
                if post_seen.insert(reloc.symbol.clone()) {
                    post_syms.push(reloc.symbol.clone());
                    added = true;
                }
            } else if pre_seen.insert(reloc.symbol.clone()) {
                pre_syms.push(reloc.symbol.clone());
                added = true;
            }
        }
        if !added {
            let pre_va = base_va + head_off as u64;
            let post_va = base_va + (text_off + post_rel) as u64;
            let pre: Vec<_> = pre_syms
                .into_iter()
                .enumerate()
                .map(|(i, s)| (s, pre_va + (i * 12) as u64))
                .collect();
            let post: Vec<_> = post_syms
                .into_iter()
                .enumerate()
                .map(|(i, s)| (s, post_va + (i * 12) as u64))
                .collect();
            let pre_vas = pre.iter().map(|(s, va)| (s.clone(), *va)).collect();
            let post_vas = post.iter().map(|(s, va)| (s.clone(), *va)).collect();
            return Ok(Veneers {
                pre_len,
                post_len,
                pre,
                post,
                pre_vas,
                post_vas,
            });
        }
    }
}

fn reloc_span(kind: RelocKind, offset: u32) -> (usize, usize) {
    let off = offset as usize;
    match kind {
        RelocKind::X64RipRel32 => (off.saturating_sub(3), off + 4),
        RelocKind::Abs32 | RelocKind::PcRel32 | RelocKind::Arm64Call26 | RelocKind::ThreadOff32 => {
            (off, off + 4)
        }
        RelocKind::Abs64 | RelocKind::Arm64AdrpAdd => (off, off + 8),
        RelocKind::Arm64Movw64 => (off, off + 16),
        RelocKind::Arm64TlvCell => (off, off + 20),
    }
}

#[derive(Clone, Copy)]
struct PatchEnv<'a> {
    text_va: u64,
    pie_arm64: bool,
    addrs: SectionAddrs,
    base_va: u64,
    shim_offs: &'a [usize],
    plan: &'a Plan<'a>,
    stub_vas: &'a [u64],
    got_vas: &'a [u64],
    veneers: &'a Veneers,
}

impl<'a> PatchEnv<'a> {
    fn resolve(&self, sym: &Symbol, vas: &[u64]) -> Result<(bool, u64)> {
        self.plan
            .resolve_with(sym, self.addrs, (self.base_va, self.shim_offs), |name| {
                Ok(vas[self.plan.import_idx(name)?])
            })
    }

    fn patch_x64_rip_rel32(&self, reloc: &CodeReloc, at: usize, out: &mut [u8]) -> Result<()> {
        if let Symbol::Word(val) = reloc.symbol {
            let val32 = u32::try_from(val).context("X64RipRel32 word exceeds u32")?;
            let reg = ((out[at - 3] & 4) << 1) | ((out[at - 1] >> 3) & 7);
            let mut bytes = [0x40 | ((reg >> 3) & 1), 0xB8 | (reg & 7), 0, 0, 0, 0, 0x90];
            bytes[2..6].copy_from_slice(&val32.to_le_bytes());
            out[at - 3..at + 4].copy_from_slice(&bytes);
            return Ok(());
        }
        let (load_got, value) = self.resolve(&reloc.symbol, self.got_vas)?;
        if load_got {
            out[at - 2] = 0x8B;
        }
        let next_ip = self.text_va + u64::from(reloc.offset) + 4;
        let delta = i32::try_from(value as i64 - next_ip as i64)
            .context("X64RipRel32 target out of range")?;
        out[at..at + 4].copy_from_slice(&delta.to_le_bytes());
        Ok(())
    }

    fn patch_abs64_got(&self, reloc: &CodeReloc, at: usize, out: &mut [u8]) -> Result<bool> {
        let (load_got, slot) = self.resolve(&reloc.symbol, self.got_vas)?;
        if !load_got {
            return Ok(false);
        }
        let next_ip = self.text_va + u64::from(reloc.offset) + 5;
        let delta = i32::try_from(slot as i64 - next_ip as i64)
            .context("Abs64 import GOT slot out of range")?;
        let reg = out[at - 1] & 7;
        out[at - 2] = 0x48 | ((out[at - 2] & 1) << 2);
        out[at - 1] = 0x8B;
        out[at] = 0x05 | (reg << 3);
        out[at + 1..at + 5].copy_from_slice(&delta.to_le_bytes());
        out[at + 5..at + 8].copy_from_slice(&[0x0F, 0x1F, 0x00]);
        Ok(true)
    }

    fn patch_one(&self, reloc: &CodeReloc, at: usize, out: &mut [u8]) -> Result<()> {
        if reloc.kind == RelocKind::Arm64AdrpAdd {
            let pc_va = self.text_va + u64::from(reloc.offset);
            let (load_got, target_va) = self.resolve(&reloc.symbol, self.got_vas)?;
            return shims::patch_arm64_adrp_add(
                &mut out[at..at + 8],
                pc_va,
                load_got,
                target_va,
                matches!(reloc.symbol, Symbol::Word(_)),
            );
        }
        if self.pie_arm64
            && reloc.kind == RelocKind::Arm64Movw64
            && !matches!(reloc.symbol, Symbol::Word(_))
        {
            let pc_va = self.text_va + u64::from(reloc.offset);
            let (load_got, target_va) = self.resolve(&reloc.symbol, self.got_vas)?;
            return shims::patch_arm64_pie(&mut out[at..at + 16], pc_va, load_got, target_va);
        }
        if reloc.kind == RelocKind::ThreadOff32 {
            return tls::patch_thread_off32(&reloc.symbol, self.addrs, &mut out[at..at + 4]);
        }
        if reloc.kind == RelocKind::Arm64TlvCell {
            let pc_va = self.text_va + u64::from(reloc.offset);
            return tls::patch_arm64_tlv_cell(
                &reloc.symbol,
                self.addrs,
                pc_va,
                &mut out[at..at + 20],
            );
        }
        if reloc.kind == RelocKind::X64RipRel32 {
            return self.patch_x64_rip_rel32(reloc, at, out);
        }
        if reloc.kind == RelocKind::Abs64
            && at >= 2
            && out[at - 2] & 0xF8 == 0x48
            && out[at - 1] & 0xF8 == 0xB8
            && self.patch_abs64_got(reloc, at, out)?
        {
            return Ok(());
        }
        let (_, value) = self.resolve(&reloc.symbol, self.stub_vas)?;
        match reloc.kind {
            RelocKind::Arm64Call26 => {
                let pc_va = self.text_va + u64::from(reloc.offset);
                let target = self
                    .veneers
                    .resolve(pc_va, value, &reloc.symbol)
                    .with_context(|| {
                        format!("Arm64Call26 target {value:#x} out of range from {pc_va:#x}")
                    })?;
                let delta = target as i64 - pc_va as i64;
                let opcode = u32::from_le_bytes([out[at], out[at + 1], out[at + 2], out[at + 3]])
                    & 0xFC00_0000;
                debug_assert!(opcode == 0x9400_0000 || opcode == 0x1400_0000);
                let word = opcode | (((delta >> 2) as u32) & 0x03FF_FFFF);
                out[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
            RelocKind::PcRel32 => {
                let next_ip = self.text_va + u64::from(reloc.offset) + 4;
                let addend = i64::from(i32::from_le_bytes([
                    out[at],
                    out[at + 1],
                    out[at + 2],
                    out[at + 3],
                ]));
                let delta = i32::try_from(value as i64 + addend - next_ip as i64)
                    .context("PcRel32 target out of range")?;
                out[at..at + 4].copy_from_slice(&delta.to_le_bytes());
            }
            kind => patch_reloc(kind, value, &mut out[at..])?,
        }
        Ok(())
    }

    fn patch_slice(self, relocs: &[&CodeReloc], slice_base: usize, out: &mut [u8]) -> Result<()> {
        for &reloc in relocs {
            let at = (reloc.offset as usize) - slice_base;
            self.patch_one(reloc, at, out)?;
        }
        Ok(())
    }
}

fn partition_relocs(sorted: &[&CodeReloc], workers: usize) -> Vec<(usize, usize, usize)> {
    let step = sorted.len().div_ceil(workers).max(1);
    let mut bounds = Vec::with_capacity(workers + 1);
    let mut pos = 0;
    while pos < sorted.len() {
        let target_end = (pos + step).min(sorted.len());
        let mut max_end = 0_usize;
        for reloc in &sorted[pos..target_end] {
            max_end = max_end.max(reloc_span(reloc.kind, reloc.offset).1);
        }
        let mut end = target_end;
        while end < sorted.len() && reloc_span(sorted[end].kind, sorted[end].offset).0 < max_end {
            max_end = max_end.max(reloc_span(sorted[end].kind, sorted[end].offset).1);
            end += 1;
        }
        let slice_start = if pos == 0 {
            0
        } else {
            reloc_span(sorted[pos].kind, sorted[pos].offset).0
        };
        bounds.push((pos, end, slice_start));
        pos = end;
    }
    bounds
}

pub(crate) fn patch_code(relocs: &[CodeReloc], ctx: &PatchCtx<'_>, out: &mut [u8]) -> Result<()> {
    let mut stub_vas = Vec::with_capacity(ctx.plan.imports.len());
    let mut got_vas = Vec::with_capacity(ctx.plan.imports.len());
    for imp in &ctx.plan.imports {
        stub_vas.push((ctx.stub_va)(imp.name)?);
        got_vas.push((ctx.got_va)(imp.name)?);
    }
    let env = PatchEnv {
        text_va: ctx.text_va,
        pie_arm64: ctx.pie_arm64,
        addrs: ctx.addrs,
        base_va: ctx.base_va,
        shim_offs: ctx.shim_offs,
        plan: ctx.plan,
        stub_vas: &stub_vas,
        got_vas: &got_vas,
        veneers: ctx.veneers,
    };
    let refs: Vec<&CodeReloc> = relocs.iter().collect();
    let workers = crate::jobs().min(refs.len() / 1024);
    if workers <= 1 {
        return env.patch_slice(&refs, 0, &mut out[ctx.text_base..]);
    }
    let mut sorted = refs;
    sorted.sort_unstable_by_key(|r| r.offset);
    let bounds = partition_relocs(&sorted, workers);
    let mut slices = Vec::with_capacity(bounds.len());
    let mut tail = &mut out[ctx.text_base..];
    let mut prev_start = 0_usize;
    for (idx, &(c_start, c_end, _)) in bounds.iter().enumerate() {
        let slice_len = if idx + 1 == bounds.len() {
            tail.len()
        } else {
            bounds[idx + 1].2 - prev_start
        };
        let (head, rest) = tail.split_at_mut(slice_len);
        slices.push((&sorted[c_start..c_end], prev_start, head));
        prev_start += slice_len;
        tail = rest;
    }
    let results: Vec<Result<()>> = crate::sync::scope(|scope| {
        let handles: Vec<_> = slices
            .into_iter()
            .map(|(chunk, base, buf)| scope.spawn(move || env.patch_slice(chunk, base, buf)))
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join().map_err(|panic| {
                    anyhow::anyhow!(
                        "AOT patch worker panicked: {}",
                        crate::sync::panic_text(&*panic)
                    )
                })?
            })
            .collect()
    });
    for res in results {
        res?;
    }
    Ok(())
}
