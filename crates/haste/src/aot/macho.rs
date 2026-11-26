use std::hash::Hasher as _;

use anyhow::Result;
use haste_codegen::{Arch, Os};
use rustc_hash::FxHasher;

use super::shims::{emit_main_trampoline, emit_shims, emit_stub, plan_shims};
use super::tls::TlsLayout;
use super::{
    Buf, ExeSymbol, Executable, MACH_BASE, MACH_PAGE, PatchCtx, Plan, SectionAddrs, Symbol,
    patch_code, plan_veneers, push_name, resolve_sym,
};

const LOCAL_IN_TEXT: u32 = 0x010E;
const EXT_IN_TEXT: u32 = 0x010F;
const LC_UUID: u32 = 0x1B;
const LC_FUNCTION_STARTS: u32 = 0x26;
const MH_HAS_TLV_DESCRIPTORS: u32 = 0x0080_0000;
const S_THREAD_LOCAL_ZEROFILL: u32 = 0x12;
const S_THREAD_LOCAL_VARIABLES: u32 = 0x13;
const TLV_DESC_SIZE: usize = 24;

struct MachMeta<'a> {
    rebase_off: usize,
    rebase: &'a [u8],
    bind_off: usize,
    bind: &'a [u8],
    export_off: usize,
    export: &'a [u8],
    func_starts_off: usize,
    func_starts: &'a [u8],
    symtab_off: usize,
    indirect_off: usize,
    strtab_off: usize,
    strtab_len: usize,
    nimports: usize,
    nlocals: usize,
    nexports: usize,
    tramp_off: usize,
    libs: &'a [String],
}

fn slice_off_len(off: usize, data: &[u8]) -> (u32, u32) {
    if data.is_empty() {
        (0, 0)
    } else {
        (off as u32, data.len() as u32)
    }
}

fn write_meta(out: &mut Buf, m: &MachMeta<'_>) -> usize {
    let (rebase_off, rebase_sz) = slice_off_len(m.rebase_off, m.rebase);
    let (bind_off, bind_sz) = slice_off_len(m.bind_off, m.bind);
    let (export_off, export_sz) = slice_off_len(m.export_off, m.export);
    out.words(&[
        0x8000_0022,
        48,
        rebase_off,
        rebase_sz,
        bind_off,
        bind_sz,
        0,
        0,
        0,
        0,
        export_off,
        export_sz,
    ]);
    out.words(&[
        2,
        24,
        m.symtab_off as u32,
        (m.nlocals + m.nexports + m.nimports) as u32,
        m.strtab_off as u32,
        m.strtab_len as u32,
    ]);
    out.words(&[
        11,
        80,
        0,
        m.nlocals as u32,
        m.nlocals as u32,
        m.nexports as u32,
        (m.nlocals + m.nexports) as u32,
        m.nimports as u32,
        0,
        0,
        0,
        0,
        0,
        0,
        m.indirect_off as u32,
        (m.nimports * 2) as u32,
        0,
        0,
        0,
        0,
    ]);
    let dylinker = b"/usr/lib/dyld\0";
    out.words(&[14, (12 + dylinker.len()).next_multiple_of(8) as u32, 12]);
    out.bytes(dylinker);
    out.align(8);
    out.words(&[LC_UUID, 24]);
    let uuid_off = out.pos();
    out.bytes(&[0; 16]);
    out.words(&[0x32, 24, 1, 0x000E_0000, 0x000E_0000, 0, 0x8000_0028, 24]);
    out.u64s(&[m.tramp_off as u64, 0]);
    out.words(&[
        LC_FUNCTION_STARTS,
        16,
        m.func_starts_off as u32,
        m.func_starts.len() as u32,
    ]);
    for lib in m.libs {
        let size = (24 + lib.len() + 1).next_multiple_of(8);
        out.words(&[12, size as u32, 24, 0, 0x0001_0000, 0x0001_0000]);
        out.bytes(lib.as_bytes());
        out.u8(0);
        out.align(8);
    }
    uuid_off
}

struct TrieNode {
    terminal: Option<u64>,
    edges: Vec<(Vec<u8>, usize)>,
}

fn trie_insert(nodes: &mut Vec<TrieNode>, node: usize, rem: &[u8], image_off: u64) {
    if rem.is_empty() {
        nodes[node].terminal = Some(image_off);
        return;
    }
    let Some(pos) = nodes[node]
        .edges
        .iter()
        .position(|(label, _)| label[0] == rem[0])
    else {
        let leaf = nodes.len();
        nodes.push(TrieNode {
            terminal: Some(image_off),
            edges: Vec::new(),
        });
        nodes[node].edges.push((rem.to_vec(), leaf));
        return;
    };
    let common = nodes[node].edges[pos]
        .0
        .iter()
        .zip(rem)
        .take_while(|(a, b)| a == b)
        .count();
    let child = nodes[node].edges[pos].1;
    if common == nodes[node].edges[pos].0.len() {
        trie_insert(nodes, child, &rem[common..], image_off);
        return;
    }
    let suffix = nodes[node].edges[pos].0[common..].to_vec();
    let mid = nodes.len();
    nodes.push(TrieNode {
        terminal: None,
        edges: vec![(suffix, child)],
    });
    nodes[node].edges[pos] = (rem[..common].to_vec(), mid);
    trie_insert(nodes, mid, &rem[common..], image_off);
}

fn build_export_trie(exports: &[&ExeSymbol], prefix: &str, text_in_seg: u64) -> Vec<u8> {
    if exports.is_empty() {
        return Vec::new();
    }
    let mut nodes = vec![TrieNode {
        terminal: None,
        edges: Vec::new(),
    }];
    for sym in exports {
        let full = format!("{prefix}{}", sym.name);
        trie_insert(
            &mut nodes,
            0,
            full.as_bytes(),
            text_in_seg + u64::from(sym.offset),
        );
    }
    let mut offsets = vec![0u64; nodes.len()];
    let mut buf = Buf::new();
    loop {
        buf.0.clear();
        let mut next = vec![0u64; nodes.len()];
        for (i, node) in nodes.iter().enumerate() {
            next[i] = buf.pos() as u64;
            if let Some(image_off) = node.terminal {
                let mut term = Buf::new();
                term.uleb(0);
                term.uleb(image_off);
                buf.uleb(term.0.len() as u64);
                buf.bytes(&term.0);
            } else {
                buf.u8(0);
            }
            buf.u8(node.edges.len() as u8);
            for (label, child) in &node.edges {
                buf.bytes(label);
                buf.u8(0);
                buf.uleb(offsets[*child]);
            }
        }
        if next == offsets {
            return buf.0;
        }
        offsets = next;
    }
}

fn build_func_starts(
    exe: &Executable,
    tramp_off: usize,
    shim_offs: &[usize],
    text_off: usize,
) -> Vec<u8> {
    let mut starts: Vec<u64> = std::iter::once(tramp_off as u64)
        .chain(shim_offs.iter().map(|&off| off as u64))
        .chain(
            exe.sorted_fn_starts()
                .into_iter()
                .map(|off| text_off as u64 + u64::from(off)),
        )
        .collect();
    starts.sort_unstable();
    starts.dedup();
    let mut buf = Buf::new();
    let mut prev = 0u64;
    for off in starts {
        buf.uleb(off - prev);
        prev = off;
    }
    buf.u8(0);
    buf.0
}

fn deterministic_uuid(bytes: &[u8]) -> [u8; 16] {
    let mut h1 = FxHasher::default();
    let mut h2 = FxHasher::default();
    h1.write_u64(0x9E37_79B9_7F4A_7C15);
    h2.write_u64(0xBF58_476D_1CE4_E5B9);
    h1.write_usize(bytes.len());
    h2.write_usize(bytes.len() ^ 0x5555_5555_5555_5555);
    h1.write(bytes);
    for chunk in bytes.chunks(8) {
        let mut word = [0u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        h2.write_u64(u64::from_le_bytes(word).swap_bytes());
    }
    let mut uuid = [0u8; 16];
    uuid[..8].copy_from_slice(&h1.finish().to_le_bytes());
    uuid[8..].copy_from_slice(&h2.finish().to_le_bytes());
    uuid[6] = (uuid[6] & 0x0F) | 0x40;
    uuid[8] = (uuid[8] & 0x3F) | 0x80;
    uuid
}

fn build_dyld_info(
    exe: &Executable,
    plan: &Plan<'_>,
    data_in_seg: u64,
    rodata_in_seg: u64,
    tlv_in_seg: Option<u64>,
    single_lib: bool,
) -> (Vec<u8>, Vec<u8>) {
    let mut rebase = Buf::new();
    let mut ptrs: Vec<u64> = exe
        .pointers
        .iter()
        .filter(|p| !matches!(p.symbol, Symbol::Word(_)))
        .map(|p| if p.in_rodata { rodata_in_seg } else { data_in_seg } + u64::from(p.offset))
        .collect();
    ptrs.sort_unstable();
    ptrs.dedup();
    if !ptrs.is_empty() {
        rebase.u8(0x11);
        ptrs.into_iter().for_each(|off| {
            rebase.u8(0x22);
            rebase.uleb(off);
            rebase.u8(0x51);
        });
        rebase.u8(0);
    }
    let mut bind = Buf::new();
    if !plan.imports.is_empty() || tlv_in_seg.is_some() {
        bind.u8(if single_lib { 0x11 } else { 0x3E });
        bind.u8(0x51);
        for (i, imp) in plan.imports.iter().enumerate() {
            let name = imp.name;
            bind.u8(0x40);
            bind.bytes(exe.target.symbol_prefix.as_bytes());
            bind.bytes(name.as_bytes());
            bind.u8(0);
            bind.u8(0x72);
            bind.uleb((i * 8) as u64);
            bind.u8(0x90);
        }
        if let Some(off) = tlv_in_seg {
            bind.u8(0x11);
            bind.u8(0x40);
            bind.bytes(exe.target.symbol_prefix.as_bytes());
            bind.bytes(b"_tlv_bootstrap");
            bind.u8(0);
            bind.u8(0x72);
            bind.uleb(off);
            bind.u8(0x90);
        }
        bind.u8(0);
    }
    (rebase.0, bind.0)
}

fn write_seg(
    out: &mut Buf,
    name: &str,
    vmaddr: u64,
    vmsize: u64,
    fileoff: u64,
    filesize: u64,
    prot: u32,
    nsects: u32,
) {
    out.words(&[0x19, 72 + nsects * 80]);
    out.fixed_name(name);
    out.u64s(&[vmaddr, vmsize, fileoff, filesize]);
    out.words(&[prot, prot, nsects, 0]);
}

fn write_sect(
    out: &mut Buf,
    sect: &str,
    seg: &str,
    addr: u64,
    size: u64,
    offset: u32,
    align: u32,
    flags: u32,
    r1: u32,
    r2: u32,
) {
    out.fixed_name(sect);
    out.fixed_name(seg);
    out.u64s(&[addr, size]);
    out.words(&[offset, align, 0, 0, flags, r1, r2, 0]);
}

pub(super) fn emit(exe: &Executable, (lang_start, user_main): (u32, u32)) -> Result<Vec<u8>> {
    let arch = exe.target.arch;
    let plan = Plan::collect(exe, &[])?;
    let (stub_size, tramp_size) = if arch == Arch::X64 { (6, 20) } else { (12, 24) };

    let mut libs = vec!["/usr/lib/libSystem.B.dylib".to_owned()];
    plan.libs
        .iter()
        .filter(|&&l| !l.contains(".so"))
        .for_each(|l| libs.push(format!("@rpath/lib{l}.dylib")));
    libs.extend(exe.extra_libs.iter().cloned());

    let tls = if arch == Arch::Arm64 {
        TlsLayout::compute(&plan, exe.thread_slots)
    } else {
        TlsLayout::default()
    };
    let has_tlv = tls.total > 0;
    let (nsects_text, nsects_data) = (
        if plan.imports.is_empty() { 1 } else { 2 },
        (if plan.imports.is_empty() { 3 } else { 4 }) + if has_tlv { 2 } else { 0 },
    );
    let dylib_cmds: usize = libs
        .iter()
        .map(|l| (24 + l.len() + 1).next_multiple_of(8))
        .sum();
    let sizeofcmds = 72
        + (72 + nsects_text * 80)
        + (72 + nsects_data * 80)
        + 72
        + 48
        + 24
        + 80
        + 32
        + 24
        + 24
        + 24
        + 16
        + dylib_cmds;
    let tramp_off = (32 + sizeofcmds + 64).next_multiple_of(16);
    let shims_off = (tramp_off + tramp_size).next_multiple_of(16);
    let probe_addrs = SectionAddrs {
        text: MACH_BASE,
        thread_key: MACH_BASE,
        tlv_desc: if has_tlv { MACH_BASE } else { 0 },
        unwind_tls: tls.unwind_off,
        thread_tls: tls.cells_off,
        frame_tls: tls.frame_off,
        unwind_table: if exe.unwind_table_off.is_some() {
            MACH_BASE
        } else {
            0
        },
        unwind_walk: MACH_BASE,
        ..SectionAddrs::default()
    };
    let (shim_offs, head_off) =
        plan_shims(arch, Os::Darwin, &plan, MACH_BASE, shims_off, probe_addrs)?;
    let post_rel = exe.text.len().next_multiple_of(16);
    let veneers = plan_veneers(exe, &plan, MACH_BASE, head_off, &shim_offs, |pre, post| {
        (head_off + pre + post_rel + post).next_multiple_of(16)
    })?;
    let text_off = head_off + veneers.pre_len;
    let stubs_off = (text_off + post_rel + veneers.post_len).next_multiple_of(16);
    let text_seg_size =
        (stubs_off + plan.imports.len() * stub_size).next_multiple_of(MACH_PAGE as usize);

    let (data_seg_off, data_seg_va) = (text_seg_size, MACH_BASE + text_seg_size as u64);
    let data_off = (data_seg_off + plan.imports.len() * 8).next_multiple_of(16);
    let rodata_off = (data_off + exe.data.len()).next_multiple_of(16);
    let dso_off = rodata_off + exe.rodata.len();
    let tlv_vars_off = (dso_off + 8).next_multiple_of(8);
    let data_file_end = if has_tlv {
        tlv_vars_off + TLV_DESC_SIZE
    } else {
        dso_off + 8
    };
    let bss_off = data_file_end.next_multiple_of(16);
    let (key_off, bss_total) = plan.bss_layout(exe.bss);
    let tbss_off = (bss_off + bss_total as usize).next_multiple_of(16);
    let data_seg_filesz = data_file_end.next_multiple_of(MACH_PAGE as usize) - data_seg_off;
    let data_seg_vmsz = (tbss_off + if has_tlv { tls.total as usize } else { 0 })
        .next_multiple_of(MACH_PAGE as usize)
        - data_seg_off;

    let bss_va = MACH_BASE + bss_off as u64;
    let rodata_va = MACH_BASE + rodata_off as u64;
    let (unwind_table, unwind_walk) = plan.unwind_addrs(exe, rodata_va, MACH_BASE, &shim_offs);
    let addrs = SectionAddrs {
        text: MACH_BASE + text_off as u64,
        rodata: rodata_va,
        data: MACH_BASE + data_off as u64,
        bss: bss_va,
        dso_handle: MACH_BASE + dso_off as u64,
        thread_key: bss_va + u64::from(key_off),
        tlv_desc: if has_tlv {
            MACH_BASE + tlv_vars_off as u64
        } else {
            0
        },
        unwind_tls: tls.unwind_off,
        thread_tls: tls.cells_off,
        frame_tls: tls.frame_off,
        unwind_table,
        unwind_walk,
    };
    let stub_va =
        |name: &str| Ok(MACH_BASE + (stubs_off + plan.import_idx(name)? * stub_size) as u64);
    let got_va = |idx: usize| MACH_BASE + (data_seg_off + idx * 8) as u64;
    let resolve = |sym: &Symbol| resolve_sym(sym, addrs, MACH_BASE, &shim_offs, &plan, &stub_va);

    let tlv_in_seg = has_tlv.then_some((tlv_vars_off - data_seg_off) as u64);
    let (rebase, bind) = build_dyld_info(
        exe,
        &plan,
        (data_off - data_seg_off) as u64,
        (rodata_off - data_seg_off) as u64,
        tlv_in_seg,
        libs.len() == 1,
    );
    let mut strtab = vec![0u8];
    let sym_offs: Vec<u32> = plan
        .imports
        .iter()
        .map(|imp| {
            let o = strtab.len() as u32;
            strtab.extend_from_slice(exe.target.symbol_prefix.as_bytes());
            strtab.extend_from_slice(imp.name.as_bytes());
            strtab.push(0);
            o
        })
        .collect();
    let (locals, exports) = exe.split_symbols();
    let main_strx = (!locals.is_empty() && !exe.symbols.iter().any(|sym| sym.name == "main"))
        .then(|| push_name(&mut strtab, &[exe.target.symbol_prefix, "main"]));
    let local_strx =
        Executable::push_prefixed_names(&locals, exe.target.symbol_prefix, &mut strtab);
    let export_strx =
        Executable::push_prefixed_names(&exports, exe.target.symbol_prefix, &mut strtab);
    let nlocals = locals.len() + usize::from(main_strx.is_some());
    let nexports = exports.len();
    let export_trie = build_export_trie(&exports, exe.target.symbol_prefix, text_off as u64);
    let func_starts = build_func_starts(exe, tramp_off, &shim_offs, text_off);
    let linkedit_off = data_seg_off + data_seg_filesz;
    let linkedit_va = data_seg_va + data_seg_vmsz as u64;
    let bind_off = (linkedit_off + rebase.len()).next_multiple_of(8);
    let export_off = (bind_off + bind.len()).next_multiple_of(8);
    let func_starts_off = (export_off + export_trie.len()).next_multiple_of(8);
    let symtab_off = (func_starts_off + func_starts.len()).next_multiple_of(8);
    let indirect_off = symtab_off + (nlocals + nexports + plan.imports.len()) * 16;
    let strtab_off = indirect_off + plan.imports.len() * 8;
    let linkedit_filesz = (strtab_off + strtab.len()) - linkedit_off;

    let mut out = Buf::new();
    let mut flags = if libs.len() == 1 {
        0x0020_0085
    } else {
        0x0020_0005
    };
    if has_tlv {
        flags |= MH_HAS_TLV_DESCRIPTORS;
    }
    out.words(&[
        0xFEED_FACF,
        if arch == Arch::Arm64 {
            0x0100_000C
        } else {
            0x0100_0007
        },
        if arch == Arch::Arm64 { 0 } else { 3 },
        2,
        12 + libs.len() as u32,
        sizeofcmds as u32,
        flags,
        0,
    ]);
    write_seg(&mut out, "__PAGEZERO", 0, MACH_BASE, 0, 0, 0, 0);
    write_seg(
        &mut out,
        "__TEXT",
        MACH_BASE,
        text_seg_size as u64,
        0,
        text_seg_size as u64,
        5,
        nsects_text as u32,
    );
    write_sect(
        &mut out,
        "__text",
        "__TEXT",
        MACH_BASE + tramp_off as u64,
        (stubs_off - tramp_off) as u64,
        tramp_off as u32,
        4,
        0x8000_0400,
        0,
        0,
    );
    if !plan.imports.is_empty() {
        write_sect(
            &mut out,
            "__stubs",
            "__TEXT",
            MACH_BASE + stubs_off as u64,
            (plan.imports.len() * stub_size) as u64,
            stubs_off as u32,
            4,
            0x8000_0408,
            plan.imports.len() as u32,
            stub_size as u32,
        );
    }
    write_seg(
        &mut out,
        "__DATA",
        data_seg_va,
        data_seg_vmsz as u64,
        data_seg_off as u64,
        data_seg_filesz as u64,
        3,
        nsects_data as u32,
    );
    if !plan.imports.is_empty() {
        write_sect(
            &mut out,
            "__got",
            "__DATA",
            data_seg_va,
            (plan.imports.len() * 8) as u64,
            data_seg_off as u32,
            3,
            6,
            0,
            0,
        );
    }
    write_sect(
        &mut out,
        "__data",
        "__DATA",
        addrs.data,
        exe.data.len() as u64,
        data_off as u32,
        4,
        0,
        0,
        0,
    );
    write_sect(
        &mut out,
        "__const",
        "__DATA",
        addrs.rodata,
        (exe.rodata.len() + 8) as u64,
        rodata_off as u32,
        4,
        0,
        0,
        0,
    );
    if has_tlv {
        write_sect(
            &mut out,
            "__thread_vars",
            "__DATA",
            addrs.tlv_desc,
            TLV_DESC_SIZE as u64,
            tlv_vars_off as u32,
            3,
            S_THREAD_LOCAL_VARIABLES,
            0,
            0,
        );
    }
    write_sect(
        &mut out,
        "__bss",
        "__DATA",
        addrs.bss,
        u64::from(bss_total),
        0,
        4,
        1,
        0,
        0,
    );
    if has_tlv {
        write_sect(
            &mut out,
            "__thread_bss",
            "__DATA",
            MACH_BASE + tbss_off as u64,
            tls.total,
            0,
            4,
            S_THREAD_LOCAL_ZEROFILL,
            0,
            0,
        );
    }
    write_seg(
        &mut out,
        "__LINKEDIT",
        linkedit_va,
        linkedit_filesz.next_multiple_of(MACH_PAGE as usize) as u64,
        linkedit_off as u64,
        linkedit_filesz as u64,
        1,
        0,
    );
    let uuid_off = write_meta(
        &mut out,
        &MachMeta {
            rebase_off: linkedit_off,
            rebase: &rebase,
            bind_off,
            bind: &bind,
            export_off,
            export: &export_trie,
            func_starts_off,
            func_starts: &func_starts,
            symtab_off,
            indirect_off,
            strtab_off,
            strtab_len: strtab.len(),
            nimports: plan.imports.len(),
            nlocals,
            nexports,
            tramp_off,
            libs: &libs,
        },
    );

    out.0.resize(tramp_off, 0);
    emit_main_trampoline(
        arch,
        MACH_BASE + tramp_off as u64,
        addrs.text + u64::from(lang_start),
        addrs.text + u64::from(user_main),
        &mut out,
    );
    out.align(16);
    emit_shims(
        arch,
        Os::Darwin,
        &plan,
        MACH_BASE,
        0,
        addrs,
        &stub_va,
        None,
        &mut out,
    )?;
    out.align(16);
    veneers.emit_pre(&resolve, &mut out)?;
    out.bytes(&exe.text);
    let resolve_got = |name: &str| Ok(got_va(plan.import_idx(name)?));
    let ctx = PatchCtx {
        text_base: text_off,
        text_va: addrs.text,
        pie_arm64: arch == Arch::Arm64,
        addrs,
        base_va: MACH_BASE,
        shim_offs: &shim_offs,
        plan: &plan,
        stub_va: &stub_va,
        got_va: &resolve_got,
        veneers: &veneers,
    };
    patch_code(&exe.relocs, &ctx, &mut out.0)?;

    out.align(16);
    veneers.emit_post(&resolve, &mut out)?;
    (0..plan.imports.len()).for_each(|i| {
        emit_stub(
            arch,
            MACH_BASE + (stubs_off + i * stub_size) as u64,
            got_va(i),
            &mut out,
        )
    });
    out.0.resize(data_off, 0);
    out.bytes(&exe.data);
    out.0.resize(rodata_off, 0);
    out.bytes(&exe.rodata);
    out.bytes(&[0; 8]);
    for ptr in &exe.pointers {
        let off = if ptr.in_rodata { rodata_off } else { data_off } + ptr.offset as usize;
        out.patch_u64(off, resolve(&ptr.symbol)?);
    }
    out.0.resize(linkedit_off, 0);
    out.bytes(&rebase);
    out.align(8);
    out.bytes(&bind);
    out.align(8);
    out.bytes(&export_trie);
    out.align(8);
    out.bytes(&func_starts);
    out.align(8);
    if let Some(strx) = main_strx {
        out.words(&[strx, LOCAL_IN_TEXT]);
        out.u64(MACH_BASE + tramp_off as u64);
    }
    for (&strx, sym) in local_strx.iter().zip(&locals) {
        out.words(&[strx, LOCAL_IN_TEXT]);
        out.u64(addrs.text + u64::from(sym.offset));
    }
    for (&strx, sym) in export_strx.iter().zip(&exports) {
        out.words(&[strx, EXT_IN_TEXT]);
        out.u64(addrs.text + u64::from(sym.offset));
    }
    sym_offs.iter().for_each(|&off| {
        out.words(&[off, 1]);
        out.u64(0);
    });
    for _ in 0..2 {
        (0..plan.imports.len()).for_each(|idx| out.u32((nlocals + nexports + idx) as u32));
    }
    out.bytes(&strtab);
    let uuid = deterministic_uuid(&out.0);
    out.0[uuid_off..uuid_off + 16].copy_from_slice(&uuid);
    Ok(out.0)
}
