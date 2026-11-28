use anyhow::Result;
use haste_codegen::{Arch, Os};

use super::shims::{emit_elf_start, emit_main_trampoline, emit_shims, emit_stub, plan_shims};
use super::tls::TlsLayout;
use super::{
    Buf, ExeSymbol, Executable, PatchCtx, Plan, SectionAddrs, Symbol, patch_code, plan_veneers,
    push_name, resolve_sym,
};

const SECTION_NAMES: &[u8] =
    b"\0.text\0.symtab\0.strtab\0.shstrtab\0.rodata\0.data\0.bss\0.dynstr\0.dynamic\0.debug_line\0";
const LOCAL_FUNC_IN_TEXT: u32 = 0x0001_0002;
const GLOBAL_FUNC_IN_TEXT: u32 = 0x0001_0012;

struct ElfTables {
    dynstr: Vec<u8>,
    needed: Vec<u32>,
    sym_offs: Vec<u32>,
    export_offs: Vec<u32>,
    buckets: Vec<u32>,
    chains: Vec<u32>,
}

pub(super) fn elf_hash(name: &[u8]) -> u32 {
    name.iter().fold(0u32, |mut h, &byte| {
        h = (h << 4).wrapping_add(u32::from(byte));
        let g = h & 0xF000_0000;
        if g != 0 {
            h ^= g >> 24;
        }
        h & !g
    })
}

fn build_tables(plan: &Plan<'_>, extra_libs: &[String], exports: &[&ExeSymbol]) -> ElfTables {
    let mut dynstr = vec![0u8];
    let mut push_str = |s: &str| push_name(&mut dynstr, &[s]);
    let mut needed = vec![push_str("libc.so.6")];
    plan.libs
        .iter()
        .filter(|&&l| l != "libc.so.6")
        .for_each(|l| needed.push(push_str(l)));
    extra_libs.iter().for_each(|l| needed.push(push_str(l)));
    let sym_offs: Vec<u32> = plan.imports.iter().map(|imp| push_str(imp.name)).collect();
    let export_offs = Executable::push_prefixed_names(exports, "", &mut dynstr);
    let nsyms = 1 + plan.imports.len() + exports.len();
    let nbucket = nsyms.next_power_of_two().max(4);
    let (mut buckets, mut chains) = (vec![0u32; nbucket], vec![0u32; nsyms]);
    let all_names = plan
        .imports
        .iter()
        .map(|imp| imp.name)
        .chain(exports.iter().map(|exp| exp.name.as_str()));
    for (i, name) in all_names.enumerate() {
        let idx = (i + 1) as u32;
        let b = (elf_hash(name.as_bytes()) as usize) % nbucket;
        chains[idx as usize] = buckets[b];
        buckets[b] = idx;
    }
    ElfTables {
        dynstr,
        needed,
        sym_offs,
        export_offs,
        buckets,
        chains,
    }
}

fn write_phdr(out: &mut Buf, kind: u32, flags: u32, [off, va, fsz, msz, align]: [u64; 5]) {
    out.words(&[kind, flags]);
    out.u64s(&[off, va, va, fsz, msz, align]);
}

fn write_shdr(
    out: &mut Buf,
    [name, kind, link, info]: [u32; 4],
    [flags, addr, off, size, align, entsize]: [u64; 6],
) {
    out.words(&[name, kind]);
    out.u64s(&[flags, addr, off, size]);
    out.words(&[link, info]);
    out.u64s(&[align, entsize]);
}

struct ElfSections {
    rx_base: u64,
    text_va: u64,
    symtab_off: usize,
    exec: (usize, usize),
    ro: (u64, usize, usize),
    rw: (u64, usize, usize),
    bss: (u64, usize, usize),
    dynstr: (usize, usize),
    dynamic: (u64, usize, usize),
}

fn write_symbols(out: &mut Buf, exe: &Executable, sec: &ElfSections, syms: &ElfSymData) {
    let has_syms = !exe.symbols.is_empty();
    let nlocals = 1 + syms.prelude.len() + syms.locals.len();
    let nsyms = if has_syms {
        nlocals + syms.globals.len()
    } else {
        0
    };
    out.0.resize(sec.symtab_off, 0);
    let mut strtab_off = sec.symtab_off;
    if has_syms {
        out.bytes(&[0; 24]);
        for &(off, size, strx) in &syms.prelude {
            out.words(&[strx, LOCAL_FUNC_IN_TEXT]);
            out.u64s(&[sec.rx_base + off as u64, size as u64]);
        }
        for &(strx, off, size) in &syms.locals {
            out.words(&[strx, LOCAL_FUNC_IN_TEXT]);
            out.u64s(&[sec.text_va + u64::from(off), u64::from(size)]);
        }
        for &(strx, off, size) in &syms.globals {
            out.words(&[strx, GLOBAL_FUNC_IN_TEXT]);
            out.u64s(&[sec.text_va + u64::from(off), u64::from(size)]);
        }
        strtab_off = out.pos();
        out.bytes(&syms.strtab);
    }
    let names_off = out.pos();
    out.bytes(SECTION_NAMES);
    let lines_off = out.pos();
    out.bytes(&syms.lines);
    out.align(8);
    out.bytes(&[0; 64]);
    write_shdr(
        out,
        [1, 1, 0, 0],
        [
            6,
            sec.rx_base + sec.exec.0 as u64,
            sec.exec.0 as u64,
            (sec.exec.1 - sec.exec.0) as u64,
            16,
            0,
        ],
    );
    if has_syms {
        write_shdr(
            out,
            [7, 2, 3, nlocals as u32],
            [0, 0, sec.symtab_off as u64, (nsyms * 24) as u64, 8, 24],
        );
        write_shdr(
            out,
            [15, 3, 0, 0],
            [0, 0, strtab_off as u64, syms.strtab.len() as u64, 1, 0],
        );
    }
    write_shdr(
        out,
        [23, 3, 0, 0],
        [0, 0, names_off as u64, SECTION_NAMES.len() as u64, 1, 0],
    );
    write_shdr(
        out,
        [33, 1, 0, 0],
        [2, sec.ro.0, sec.ro.1 as u64, sec.ro.2 as u64, 16, 0],
    );
    write_shdr(
        out,
        [41, 1, 0, 0],
        [3, sec.rw.0, sec.rw.1 as u64, sec.rw.2 as u64, 8, 0],
    );
    write_shdr(
        out,
        [47, 8, 0, 0],
        [3, sec.bss.0, sec.bss.1 as u64, sec.bss.2 as u64, 16, 0],
    );
    write_shdr(
        out,
        [52, 3, 0, 0],
        [
            2,
            sec.dynstr.0 as u64,
            sec.dynstr.0 as u64,
            sec.dynstr.1 as u64,
            1,
            0,
        ],
    );
    let dyn_link = if has_syms { 8 } else { 6 };
    write_shdr(
        out,
        [60, 6, dyn_link, 0],
        [
            3,
            sec.dynamic.0,
            sec.dynamic.1 as u64,
            sec.dynamic.2 as u64,
            8,
            16,
        ],
    );
    if !syms.lines.is_empty() {
        write_shdr(
            out,
            [69, 1, 0, 0],
            [0, 0, lines_off as u64, syms.lines.len() as u64, 1, 0],
        );
    }
}

struct ArchLayout {
    interp: &'static [u8],
    page: usize,
    stub_size: usize,
    start_size: usize,
    tramp_size: usize,
    machine: u16,
}

fn arch_layout(arch: Arch) -> Result<ArchLayout> {
    Ok(match arch {
        Arch::X64 => ArchLayout {
            interp: b"/lib64/ld-linux-x86-64.so.2\0",
            page: 0x1000,
            stub_size: 6,
            start_size: 34,
            tramp_size: 20,
            machine: 62,
        },
        Arch::Arm64 => ArchLayout {
            interp: b"/lib/ld-linux-aarch64.so.1\0",
            page: 0x1_0000,
            stub_size: 12,
            start_size: 56,
            tramp_size: 24,
            machine: 183,
        },
    })
}

fn elf_tls_addrs(arch: Arch, plan: &Plan<'_>, thread_slots: u32) -> (u64, u64, u64, u64) {
    if arch != Arch::X64 {
        return (0, 0, 0, 0);
    }
    let tls = TlsLayout::compute(plan, thread_slots, true);
    if tls.total == 0 {
        return (0, 0, 0, 0);
    }
    let unwind_tls = if plan.has_unwind() {
        tls.total - tls.unwind_off
    } else {
        0
    };
    let frame_tls = tls.total - tls.frame_off;
    let thread_tls = tls.total - tls.cells_off;
    (tls.total, unwind_tls, thread_tls, frame_tls)
}

fn build_prelude(
    plan: &Plan<'_>,
    shim_offs: &[usize],
    [
        stubs_off,
        stub_size,
        start_off,
        start_size,
        tramp_off,
        tramp_size,
        head_off,
    ]: [usize; 7],
    strtab: &mut Vec<u8>,
) -> Vec<(usize, usize, u32)> {
    let mut prelude: Vec<(usize, usize, u32)> = plan
        .imports
        .iter()
        .enumerate()
        .map(|(i, imp)| {
            (
                stubs_off + i * stub_size,
                stub_size,
                push_name(strtab, &[imp.name, "@plt"]),
            )
        })
        .collect();
    prelude.push((start_off, start_size, push_name(strtab, &["_start"])));
    prelude.push((
        tramp_off,
        tramp_size,
        push_name(strtab, &["haste_main_trampoline"]),
    ));
    for (i, (shim, &off)) in plan.shims.iter().zip(shim_offs).enumerate() {
        let end = shim_offs.get(i + 1).copied().unwrap_or(head_off);
        prelude.push((
            off,
            end - off,
            push_name(strtab, &["haste_", &format!("{shim:?}")]),
        ));
    }
    prelude
}

struct HeaderSpec<'a> {
    arch: Arch,
    machine: u16,
    page: usize,
    phnum: usize,
    shstrndx: u16,
    shnum: u16,
    shdr_off: usize,
    entry_va: u64,
    interp: &'a [u8],
    interp_off: usize,
    ro_end: usize,
    rx: (u64, usize, usize),
    rw: (u64, usize, usize, usize),
    dyn_seg: (usize, usize),
    relro_size: usize,
    tls_seg: (u64, usize, u64),
    tables: &'a ElfTables,
    text_va: u64,
    export_spans: &'a [(u32, u32)],
    rel_ptrs: &'a [(u64, u64)],
}

fn write_ro_prefix(out: &mut Buf, spec: &HeaderSpec<'_>, got_va: impl Fn(usize) -> u64) {
    out.bytes(b"\x7FELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00");
    out.u16(3);
    out.u16(spec.machine);
    out.u32(1);
    out.u64s(&[spec.entry_va, 64, spec.shdr_off as u64]);
    out.u32(0);
    [64_u16, 56, spec.phnum as u16, 64, spec.shnum, spec.shstrndx]
        .into_iter()
        .for_each(|v| out.u16(v));
    let (phsz, ilen, pg) = (
        (spec.phnum * 56) as u64,
        spec.interp.len() as u64,
        spec.page as u64,
    );
    let (rx_sz, ro_end) = ((spec.rx.2 - spec.rx.1) as u64, spec.ro_end as u64);
    let (dyn_va, dyn_sz) = (
        spec.rw.0 + (spec.dyn_seg.0 - spec.rw.1) as u64,
        spec.dyn_seg.1 as u64,
    );
    let (rw_off, relro) = (spec.rw.1 as u64, spec.relro_size as u64);
    write_phdr(out, 6, 4, [64, 64, phsz, phsz, 8]);
    write_phdr(
        out,
        3,
        4,
        [
            spec.interp_off as u64,
            spec.interp_off as u64,
            ilen,
            ilen,
            1,
        ],
    );
    write_phdr(out, 1, 4, [0, 0, ro_end, ro_end, pg]);
    write_phdr(
        out,
        1,
        5,
        [
            spec.rx.1 as u64,
            spec.rx.0 + spec.rx.1 as u64,
            rx_sz,
            rx_sz,
            pg,
        ],
    );
    write_phdr(
        out,
        1,
        6,
        [rw_off, spec.rw.0, spec.rw.2 as u64, spec.rw.3 as u64, pg],
    );
    write_phdr(
        out,
        2,
        6,
        [spec.dyn_seg.0 as u64, dyn_va, dyn_sz, dyn_sz, 8],
    );
    write_phdr(out, 0x6474_E552, 4, [rw_off, spec.rw.0, relro, relro, 1]);
    if spec.tls_seg.2 != 0 {
        write_phdr(
            out,
            7,
            4,
            [spec.tls_seg.1 as u64, spec.tls_seg.0, 0, spec.tls_seg.2, 16],
        );
    }
    let nsyms = spec.tables.chains.len();
    out.bytes(spec.interp);
    out.align(8);
    out.words(&[spec.tables.buckets.len() as u32, nsyms as u32]);
    out.words(&spec.tables.buckets);
    out.words(&spec.tables.chains);
    out.align(8);
    out.bytes(&[0; 24]);
    spec.tables.sym_offs.iter().for_each(|&off| {
        out.words(&[off, 0x12]);
        out.u64s(&[0, 0]);
    });
    for (&off, &(fn_off, fn_size)) in spec.tables.export_offs.iter().zip(spec.export_spans) {
        out.words(&[off, GLOBAL_FUNC_IN_TEXT]);
        out.u64s(&[spec.text_va + u64::from(fn_off), u64::from(fn_size)]);
    }
    out.bytes(&spec.tables.dynstr);
    out.align(8);
    let (glob_type, rel_type) = if spec.arch == Arch::X64 {
        (6_u64, 8_u64)
    } else {
        (1025_u64, 1027_u64)
    };
    (0..spec.tables.sym_offs.len())
        .for_each(|i| out.u64s(&[got_va(i), ((i as u64 + 1) << 32) | glob_type, 0]));
    spec.rel_ptrs
        .iter()
        .for_each(|&(slot, target)| out.u64s(&[slot, rel_type, target]));
}

struct EmitBody<'a> {
    arch: Arch,
    rx_base: u64,
    stub_size: usize,
    offs: [usize; 6],
    entry: (u32, u32),
    addrs: SectionAddrs,
    shim_offs: &'a [usize],
    plan: &'a Plan<'a>,
    veneers: &'a super::Veneers,
    tables: &'a ElfTables,
    dyn_meta: [u64; 6],
}

fn write_rx_and_rw(out: &mut Buf, exe: &Executable, body: &EmitBody<'_>) -> Result<()> {
    let [stubs_off, start_off, tramp_off, text_off, rw_off, data_off] = body.offs;
    let [got_off, dyn_off, hash_va, dynstr_va, dynsym_va, nrela] = body.dyn_meta;
    let stub_va = |name: &str| {
        Ok(body.rx_base + (stubs_off + body.plan.import_idx(name)? * body.stub_size) as u64)
    };
    let got_va = |idx: usize| body.addrs.rodata + ((got_off as usize - rw_off) + idx * 8) as u64;
    let resolve = |sym: &Symbol| {
        resolve_sym(
            sym,
            body.addrs,
            body.rx_base,
            body.shim_offs,
            body.plan,
            &stub_va,
        )
    };
    out.align(16);
    (0..body.plan.imports.len()).for_each(|i| {
        emit_stub(
            body.arch,
            body.rx_base + (stubs_off + i * body.stub_size) as u64,
            got_va(i),
            out,
        )
    });
    out.align(16);
    emit_elf_start(
        body.arch,
        body.rx_base + start_off as u64,
        body.rx_base + tramp_off as u64,
        got_va(body.plan.import_idx("__libc_start_main")?),
        out,
    );
    out.align(16);
    emit_main_trampoline(
        body.arch,
        body.rx_base + tramp_off as u64,
        body.addrs.text + u64::from(body.entry.0),
        body.addrs.text + u64::from(body.entry.1),
        out,
    );
    out.align(16);
    emit_shims(
        body.arch,
        Os::Linux,
        body.plan,
        body.rx_base,
        0,
        body.addrs,
        &stub_va,
        None,
        out,
    )?;
    out.align(16);
    body.veneers.emit_pre(&resolve, out)?;
    out.bytes(&exe.text);
    out.align(16);
    body.veneers.emit_post(&resolve, out)?;
    let resolve_got = |name: &str| Ok(got_va(body.plan.import_idx(name)?));
    let ctx = PatchCtx {
        text_base: text_off,
        text_va: body.addrs.text,
        pie_arm64: body.arch == Arch::Arm64,
        pie_x64: body.arch == Arch::X64,
        addrs: body.addrs,
        base_va: body.rx_base,
        shim_offs: body.shim_offs,
        plan: body.plan,
        stub_va: &stub_va,
        got_va: &resolve_got,
        veneers: body.veneers,
    };
    patch_code(&exe.relocs, &ctx, &mut out.0)?;
    out.0.resize(rw_off, 0);
    out.bytes(&exe.rodata);
    out.bytes(&[0; 8]);
    for ptr in exe.pointers.iter().filter(|ptr| ptr.in_rodata) {
        out.patch_u64(rw_off + ptr.offset as usize, resolve(&ptr.symbol)?);
    }
    out.0.resize(dyn_off as usize, 0);
    body.tables.needed.iter().for_each(|&off| {
        out.i64(1);
        out.u64(u64::from(off));
    });
    let rela_off_va = (dynstr_va as usize + body.tables.dynstr.len()).next_multiple_of(8) as u64;
    for (tag, val) in [
        (4_i64, hash_va),
        (5, dynstr_va),
        (6, dynsym_va),
        (7, rela_off_va),
        (8, nrela * 24),
        (9, 24),
        (10, body.tables.dynstr.len() as u64),
        (11, 24),
        (30, 8),
        (0x6FFF_FFFB, 0x0800_0001),
        (0, 0),
    ] {
        out.i64(tag);
        out.u64(val);
    }
    out.0.resize(data_off, 0);
    out.bytes(&exe.data);
    for ptr in exe.pointers.iter().filter(|ptr| !ptr.in_rodata) {
        out.patch_u64(data_off + ptr.offset as usize, resolve(&ptr.symbol)?);
    }
    Ok(())
}

struct ElfSymData {
    strtab: Vec<u8>,
    locals: Vec<(u32, u32, u32)>,
    globals: Vec<(u32, u32, u32)>,
    prelude: Vec<(usize, usize, u32)>,
    lines: Vec<u8>,
    shdr_off: usize,
    shstrndx: u16,
    shnum: u16,
}

fn build_optional_syms(
    exe: &Executable,
    plan: &Plan,
    shim_offs: &[usize],
    offs: [usize; 7],
    (symtab_off, text_va, fn_starts): (usize, u64, &[u32]),
) -> ElfSymData {
    let has_syms = !exe.symbols.is_empty();
    let mut strtab = Vec::new();
    let (locals, globals, prelude) = if has_syms {
        strtab.push(0);
        let (local_syms, export_syms) = exe.split_symbols();
        let local_strx = Executable::push_prefixed_names(&local_syms, "", &mut strtab);
        let export_strx = Executable::push_prefixed_names(&export_syms, "", &mut strtab);
        let span = |sym: &ExeSymbol| (sym.offset, exe.fn_size(fn_starts, sym.offset));
        let locals = local_strx
            .into_iter()
            .zip(&local_syms)
            .map(|(strx, s)| {
                let (off, sz) = span(s);
                (strx, off, sz)
            })
            .collect();
        let globals = export_strx
            .into_iter()
            .zip(&export_syms)
            .map(|(strx, s)| {
                let (off, sz) = span(s);
                (strx, off, sz)
            })
            .collect();
        (
            locals,
            globals,
            build_prelude(plan, shim_offs, offs, &mut strtab),
        )
    } else {
        (Vec::new(), Vec::new(), Vec::new())
    };
    let want_lines = !exe.sources.is_empty() || !exe.lines.is_empty();
    let lines = if want_lines {
        super::dwarf::debug_line(text_va, exe.text.len() as u32, &exe.lines, &exe.sources)
    } else {
        Vec::new()
    };
    let sym_bytes = if has_syms {
        (1 + prelude.len() + locals.len() + globals.len()) * 24 + strtab.len()
    } else {
        0
    };
    let shdr_off = (symtab_off + sym_bytes + SECTION_NAMES.len() + lines.len()).next_multiple_of(8);
    let shstrndx: u16 = if has_syms { 4 } else { 2 };
    let shnum: u16 = if has_syms { 10 } else { 8 } + u16::from(!lines.is_empty());
    ElfSymData {
        strtab,
        locals,
        globals,
        prelude,
        lines,
        shdr_off,
        shstrndx,
        shnum,
    }
}

pub(super) fn emit(exe: &Executable, (lang_start, user_main): (u32, u32)) -> Result<Vec<u8>> {
    let arch = exe.target.arch;
    let plan = Plan::collect(exe, &["__libc_start_main"])?;
    let ArchLayout {
        interp,
        page,
        stub_size,
        start_size,
        tramp_size,
        machine,
    } = arch_layout(arch)?;
    let (_, exports) = exe.split_symbols();
    let fn_starts = exe.sorted_fn_starts();
    let export_spans: Vec<(u32, u32)> = exports
        .iter()
        .map(|s| (s.offset, exe.fn_size(&fn_starts, s.offset)))
        .collect();
    let (rx_base, tables, nsyms) = (
        page as u64,
        build_tables(&plan, &exe.extra_libs, &exports),
        1 + plan.imports.len() + exports.len(),
    );
    let (tls, unwind_tls, thread_tls, frame_tls) = elf_tls_addrs(arch, &plan, exe.thread_slots);
    let phnum = if tls == 0 { 7 } else { 8 };
    let interp_off = 64 + phnum * 56;
    let hash_off = (interp_off + interp.len()).next_multiple_of(8);
    let dynsym_off = (hash_off + (2 + tables.buckets.len() + nsyms) * 4).next_multiple_of(8);
    let dynstr_off = dynsym_off + nsyms * 24;
    let rela_off = (dynstr_off + tables.dynstr.len()).next_multiple_of(8);
    let nrel_ptrs = exe
        .pointers
        .iter()
        .filter(|p| !matches!(p.symbol, Symbol::Word(_)))
        .count();
    let nrela = plan.imports.len() + nrel_ptrs;
    let ro_end = rela_off + nrela * 24;
    let stubs_off = ro_end.next_multiple_of(16);
    let start_off = (stubs_off + plan.imports.len() * stub_size).next_multiple_of(16);
    let tramp_off = (start_off + start_size).next_multiple_of(16);
    let shims_off = (tramp_off + tramp_size).next_multiple_of(16);
    let probe_addrs = SectionAddrs {
        text: rx_base,
        thread_key: rx_base,
        unwind_tls,
        thread_tls,
        frame_tls,
        unwind_table: if exe.unwind_table_off.is_some() {
            rx_base
        } else {
            0
        },
        unwind_walk: rx_base,
        ..SectionAddrs::default()
    };
    let (shim_offs, head_off) =
        plan_shims(arch, Os::Linux, &plan, rx_base, shims_off, probe_addrs)?;
    let veneers = plan_veneers(exe, &plan, rx_base, head_off, &shim_offs, |_, _| stubs_off)?;
    let text_off = head_off + veneers.pre_len;
    let rw_off =
        (text_off + exe.text.len().next_multiple_of(16) + veneers.post_len).next_multiple_of(16);
    let rw_va = (rw_off + 2 * page) as u64;
    let dso_off = rw_off + exe.rodata.len();
    let got_off = (dso_off + 8).next_multiple_of(8);
    let (dyn_off, dyn_size) = (
        got_off + plan.imports.len() * 8,
        (tables.needed.len() + 11) * 16,
    );
    let data_off = (dyn_off + dyn_size).next_multiple_of(page);
    let relro_size = data_off - rw_off;
    let data_end = data_off + exe.data.len();
    let (rw_filesz, symtab_off) = (data_end - rw_off, data_end.next_multiple_of(8));
    let (bss_in_rw, (key_off, bss_total)) =
        (rw_filesz.next_multiple_of(16), plan.bss_layout(exe.bss));
    let prelude_offs = [
        stubs_off, stub_size, start_off, start_size, tramp_off, tramp_size, head_off,
    ];
    let syms = build_optional_syms(
        exe,
        &plan,
        &shim_offs,
        prelude_offs,
        (symtab_off, rx_base + text_off as u64, &fn_starts),
    );
    let (rodata_va, data_va, bss_va) = (rw_va, rw_va + relro_size as u64, rw_va + bss_in_rw as u64);
    let (unwind_table, unwind_walk) = plan.unwind_addrs(exe, rodata_va, rx_base, &shim_offs);
    let addrs = SectionAddrs {
        text: rx_base + text_off as u64,
        rodata: rodata_va,
        data: data_va,
        bss: bss_va,
        dso_handle: rw_va + (dso_off - rw_off) as u64,
        thread_key: bss_va + u64::from(key_off),
        tlv_desc: 0,
        unwind_tls,
        thread_tls,
        frame_tls,
        unwind_table,
        unwind_walk,
    };
    let stub_va =
        |name: &str| Ok(rx_base + (stubs_off + plan.import_idx(name)? * stub_size) as u64);
    let mut rel_ptrs = Vec::with_capacity(nrel_ptrs);
    for ptr in exe
        .pointers
        .iter()
        .filter(|p| !matches!(p.symbol, Symbol::Word(_)))
    {
        let slot = if ptr.in_rodata { rodata_va } else { data_va } + u64::from(ptr.offset);
        let target = resolve_sym(&ptr.symbol, addrs, rx_base, &shim_offs, &plan, &stub_va)?;
        rel_ptrs.push((slot, target));
    }
    let mut out = Buf::new();
    let hdr = HeaderSpec {
        arch,
        machine,
        page,
        phnum,
        shstrndx: syms.shstrndx,
        shnum: syms.shnum,
        shdr_off: syms.shdr_off,
        entry_va: rx_base + start_off as u64,
        interp,
        interp_off,
        ro_end,
        rx: (rx_base, stubs_off, rw_off),
        rw: (rw_va, rw_off, rw_filesz, bss_in_rw + bss_total as usize),
        dyn_seg: (dyn_off, dyn_size),
        relro_size,
        tls_seg: (bss_va, rw_off + bss_in_rw, tls),
        tables: &tables,
        text_va: addrs.text,
        export_spans: &export_spans,
        rel_ptrs: &rel_ptrs,
    };
    write_ro_prefix(&mut out, &hdr, |idx| {
        rw_va + ((got_off - rw_off) + idx * 8) as u64
    });
    let body = EmitBody {
        arch,
        rx_base,
        stub_size,
        offs: [stubs_off, start_off, tramp_off, text_off, rw_off, data_off],
        entry: (lang_start, user_main),
        addrs,
        shim_offs: &shim_offs,
        plan: &plan,
        veneers: &veneers,
        tables: &tables,
        dyn_meta: [
            got_off as u64,
            dyn_off as u64,
            hash_off as u64,
            dynstr_off as u64,
            dynsym_off as u64,
            nrela as u64,
        ],
    };
    write_rx_and_rw(&mut out, exe, &body)?;
    let sec = ElfSections {
        rx_base,
        text_va: addrs.text,
        symtab_off,
        exec: (stubs_off, rw_off),
        ro: (rodata_va, rw_off, exe.rodata.len()),
        rw: (data_va, data_off, exe.data.len()),
        bss: (bss_va, rw_off + bss_in_rw, bss_total as usize),
        dynstr: (dynstr_off, tables.dynstr.len()),
        dynamic: (rw_va + (dyn_off - rw_off) as u64, dyn_off, dyn_size),
    };
    write_symbols(&mut out, exe, &sec, &syms);
    Ok(out.0)
}
