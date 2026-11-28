use super::*;
use haste_codegen::{CodeUnwindDrop, RelocKind, Site, StencilReloc, UnwindArg};

use crate::graph::cfg::Target as CfgTarget;
use crate::host::HostSymbol;
use crate::image::aot::AotCode;
use crate::lower::{
    RodataId, RodataReloc, RodataTarget, SymbolSource, VtableEntry, VtableId, Word,
};
use crate::names::Primitive;
use crate::outline::scan::scan;
use crate::outline::source::Source;
use crate::sema::ty::{Kind, Types};

use codec::decode_code_blob as decode_code;
use codec::encode_code_blob as encode_code;

fn sample_aot_code() -> AotCode {
    AotCode {
        bytes: vec![0x90, 0xc3, 0x48, 0x89, 0xe5],
        sites: vec![Site {
            ret: 2,
            handler: 4,
            frame_offset: 32,
        }],
        relocs: vec![
            StencilReloc {
                offset: 1,
                symbol: 0,
                kind: RelocKind::PcRel32,
            },
            StencilReloc {
                offset: 3,
                symbol: 0x8000_0000,
                kind: RelocKind::Arm64Call26,
            },
        ],
        loops: vec![1..4],
        lines: vec![(0, 0x1122_3344_0000_0010), (2, 0x1122_3344_0000_0020)],
        unwind_drops: vec![CodeUnwindDrop {
            local_off: -16,
            flag_off: -17,
            drop_sym: 1,
            arg: UnwindArg::Indirect,
        }],
        saved_regs: 0x001f,
    }
}

pub(super) fn sample_config(flags_bits: u64, target_hash: u64, triple_hash: u64) -> AotConfigKey {
    let dict_cfg = (512, 8);
    let mut h = FxHasher::default();
    CACHE_MAGIC.hash(&mut h);
    CACHE_VERSION.hash(&mut h);
    target_hash.hash(&mut h);
    triple_hash.hash(&mut h);
    flags_bits.hash(&mut h);
    0u64.hash(&mut h);
    dict_cfg.hash(&mut h);
    AotConfigKey {
        target_hash,
        triple_hash,
        flags_bits,
        global_alloc_stamp: 0,
        dict_cfg,
        config_digest: h.finish(),
    }
}

#[test]
fn aot_code_encode_decode_round_trip() {
    let code = sample_aot_code();
    let encoded = encode_code(&code).expect("encode");
    let decoded = decode_code(&encoded).expect("decode");
    assert_eq!(decoded.bytes(), code.bytes());
    assert_eq!(decoded.sites(), code.sites());
    assert_eq!(decoded.relocs(), code.relocs());
    assert_eq!(decoded.loops(), code.loops());
    assert_eq!(decoded.lines(), code.lines());
    assert_eq!(decoded.unwind_drops(), code.unwind_drops());
    assert_eq!(decoded.saved_regs(), code.saved_regs());
}

#[test]
fn bundle_round_trip_and_key_invalidation() {
    let tmp = std::env::temp_dir().join(format!("haste-aot-cache-test-{}", std::process::id()));
    drop(std::fs::remove_dir_all(&tmp));
    std::fs::create_dir_all(&tmp).expect("mkdir");
    let path = tmp.join("bundle.aot");
    let code = Arc::new(sample_aot_code());
    let base_cfg = sample_config(0b0010_0010, 0x5555_6666, 0x7777_8888);
    let item_key = 0x1234_5678;
    let substs_hash = 0x9999_1111;
    let stamps_digest = 0xaaaa_2222;
    let flags = 0b10u32;
    let (key_hi, key_lo) = hash_entry_128(
        base_cfg.config_digest,
        item_key,
        substs_hash,
        stamps_digest,
        flags,
    );
    let code_blob = encode_code(&code).expect("encode code");
    let mut payload = Vec::new();
    for w in [item_key, substs_hash, stamps_digest] {
        payload.extend_from_slice(&w.to_le_bytes());
    }
    payload.extend_from_slice(&flags.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    payload.extend_from_slice(&0u32.to_le_bytes());
    let sections = tables::BundleSections {
        items_bytes: Vec::new(),
        items_count: 0,
        tys_bytes: Vec::new(),
        tys_count: 0,
        rodatas: Vec::new(),
        vtables: Vec::new(),
        codes: vec![tables::CodeEntrySpan {
            code_off: 0,
            code_len: code_blob.len() as u32,
        }],
        records: vec![(key_hi, key_lo, code_blob.len() as u32, payload.len() as u32)],
        blob: [code_blob, payload].concat(),
    };
    write_bundle(&path, CACHE_MAGIC, CACHE_VERSION, base_cfg, &sections).expect("write");

    let (backing, hdr) =
        open_bundle(&path, CACHE_MAGIC, CACHE_VERSION, base_cfg).expect("open valid");
    assert!(
        find_record(
            backing.as_slice(),
            hdr.records_off,
            hdr.records_count,
            (key_hi, key_lo),
        )
        .is_some()
    );
    let diff_triple = sample_config(base_cfg.flags_bits, base_cfg.target_hash, 0x9999_0000);
    assert!(open_bundle(&path, CACHE_MAGIC, CACHE_VERSION, diff_triple).is_none());
    let diff_cpu = sample_config(
        base_cfg.flags_bits ^ 1,
        base_cfg.target_hash,
        base_cfg.triple_hash,
    );
    assert!(open_bundle(&path, CACHE_MAGIC, CACHE_VERSION, diff_cpu).is_none());
    let diff_profile = sample_config(
        base_cfg.flags_bits ^ (1 << 1),
        base_cfg.target_hash,
        base_cfg.triple_hash,
    );
    assert!(open_bundle(&path, CACHE_MAGIC, CACHE_VERSION, diff_profile).is_none());
    let diff_unwind = sample_config(
        base_cfg.flags_bits ^ (1 << 2),
        base_cfg.target_hash,
        base_cfg.triple_hash,
    );
    assert!(open_bundle(&path, CACHE_MAGIC, CACHE_VERSION, diff_unwind).is_none());
    let diff_stamp = sample_config(base_cfg.flags_bits, 0x1111_2222, base_cfg.triple_hash);
    assert!(open_bundle(&path, CACHE_MAGIC, CACHE_VERSION, diff_stamp).is_none());

    let (bad_hi, bad_lo) = hash_entry_128(
        base_cfg.config_digest,
        item_key,
        substs_hash ^ 1,
        stamps_digest,
        flags,
    );
    assert!(
        find_record(
            backing.as_slice(),
            hdr.records_off,
            hdr.records_count,
            (bad_hi, bad_lo),
        )
        .is_none()
    );

    let mut bad_magic = backing.as_slice().to_vec();
    bad_magic[0..4].copy_from_slice(b"NOPE");
    assert!(parse_bundle_header(&bad_magic, CACHE_MAGIC, CACHE_VERSION, base_cfg).is_none());
    let mut bad_ver = backing.as_slice().to_vec();
    bad_ver[8..12].copy_from_slice(&999u32.to_le_bytes());
    assert!(parse_bundle_header(&bad_ver, CACHE_MAGIC, CACHE_VERSION, base_cfg).is_none());
    drop(backing);
    drop(std::fs::remove_dir_all(&tmp));
}

#[test]
fn bundle_stale_prune_keeps_newest() {
    let tmp = std::env::temp_dir().join(format!("haste-aot-prune-test-{}", std::process::id()));
    drop(std::fs::remove_dir_all(&tmp));
    std::fs::create_dir_all(&tmp).expect("mkdir");
    let a = tmp.join("a.aot");
    let b = tmp.join("b.aot");
    let c = tmp.join("c.aot");
    std::fs::write(&a, b"a").expect("write a");
    std::thread::sleep(std::time::Duration::from_millis(20));
    std::fs::write(&b, b"b").expect("write b");
    std::thread::sleep(std::time::Duration::from_millis(20));
    std::fs::write(&c, b"c").expect("write c");
    prune_stale_bundles(&tmp, &a, "aot", 1);
    assert!(a.exists());
    assert!(c.exists());
    assert!(!b.exists());
    drop(std::fs::remove_dir_all(&tmp));
}

#[test]
fn shared_tables_and_symbol_sources_round_trip() {
    struct DummySlots {
        rodatas: std::sync::Mutex<Vec<(Vec<u8>, Vec<RodataReloc>)>>,
        vtables: std::sync::Mutex<Vec<Vec<VtableEntry>>>,
    }
    impl Slots for DummySlots {
        fn entry(&self, index: usize) -> i64 {
            index as i64
        }
        fn data(&self, _: &[u8]) -> i64 {
            0
        }
        fn zeroed(&self, _: usize) -> i64 {
            0
        }
        fn table(&self, _: &[Word]) -> i64 {
            0
        }
        fn cell(&self, _: usize, _: u32) -> i64 {
            0
        }
        fn want(&self, _: crate::sema::reach::Instance) {}
        fn wanted_len(&self) -> usize {
            0
        }
        fn bodies(&self) -> &crate::lower::Bodies {
            unreachable!("unused")
        }
        fn allow_inline(&self) -> bool {
            true
        }
        fn low_entries(&self) -> bool {
            true
        }
        fn rodata(&self, bytes: &[u8], relocs: &[RodataReloc]) -> RodataId {
            let mut g = crate::sync::lock(&self.rodatas);
            let id = RodataId(g.len() as u32);
            g.push((bytes.to_vec(), relocs.to_vec()));
            id
        }
        fn static_rodata(&self, _: usize, _: &[u8]) -> RodataId {
            RodataId(0)
        }
        fn vtable_id(&self, entries: &[VtableEntry]) -> VtableId {
            let mut g = crate::sync::lock(&self.vtables);
            let id = VtableId(g.len() as u32);
            g.push(entries.to_vec());
            id
        }
        fn vtable_entries(&self, _: VtableId) -> Option<Vec<VtableEntry>> {
            None
        }
        fn resolve_symbol(&self, _: &SymbolSource) -> u64 {
            0
        }
    }

    let tmp = std::env::temp_dir().join(format!("haste-aot-tables-test-{}", std::process::id()));
    drop(std::fs::remove_dir_all(&tmp));
    std::fs::create_dir_all(&tmp).expect("mkdir");
    let src = tmp.join("main.rs");
    let helper_path = tmp.join("helper.rs");
    std::fs::write(&src, "fn main() {}\n").expect("write");
    std::fs::write(&helper_path, "fn helper() {}\n").expect("write helper");
    let mut outline = Outline::file(&src, CfgTarget::default());
    let main_src = Source::parse(src.clone(), "fn main() {}\n".to_owned()).expect("parse main");
    let main_items = scan(&main_src).expect("scan main");
    let helper_src =
        Source::parse(helper_path.clone(), "fn helper() {}\n".to_owned()).expect("parse");
    let helper_items = scan(&helper_src).expect("scan");
    outline.crates[0]
        .files
        .push(Arc::new(crate::outline::tree::File::new(
            main_src, main_items, false,
        )));
    outline.crates[0]
        .files
        .push(Arc::new(crate::outline::tree::File::new(
            helper_src,
            helper_items,
            false,
        )));
    let item = ItemRef {
        krate: 0,
        file: 0,
        item: 0,
    };
    let types = Types::default();
    let u64_ty = types.intern(Kind::Prim(Primitive::U64));
    let substs = types.intern_substs(&[u64_ty]);
    let mut reached = Reached::default();
    reached.list.push((
        (item, substs),
        Ok(Arc::new(crate::sema::infer::Typed::default())),
    ));
    reached.index.insert((item, substs), 0);
    let stamps = Stamps::compute(&outline, 0, 1);
    let mut state = AotState::new(1).into_inner().expect("lock");
    let child_ro = state.alloc_rodata(b"hi", &[]);
    let parent_ro = state.alloc_rodata(
        b"ptr",
        &[
            RodataReloc {
                offset: 0,
                target: RodataTarget::Rodata(child_ro),
            },
            RodataReloc {
                offset: 8,
                target: RodataTarget::Fn(0),
            },
        ],
    );
    let vt = state.intern_vtable(&[
        VtableEntry::Word(Word::Int(16)),
        VtableEntry::Word(Word::Code(0)),
    ]);
    let code = Arc::new(sample_aot_code());
    let relocs = vec![
        SymbolSource::Slot(0),
        SymbolSource::Rodata(parent_ro),
        SymbolSource::Vtable(vt),
        SymbolSource::StaticCell { index: 0, size: 8 },
        SymbolSource::PromotedCell {
            owner: 0,
            seq: 2,
            size: 16,
        },
        SymbolSource::Host(HostSymbol::Alloc),
        SymbolSource::Word(42),
        SymbolSource::ThreadSlot(0),
    ];
    let cfg = sample_config(0, 1, 2);
    let item_key = stable_entry_item_key(&outline, &stamps, item).expect("item key");
    let mut memo = HashMap::default();
    let sh = stable_substs_hash(&outline, &types, &stamps, substs, &mut memo).expect("sh");
    let sd = touched_stamps_digest(&types, &stamps, item, substs).expect("sd");
    let (key_hi, key_lo) = hash_entry_128(cfg.config_digest, item_key, sh, sd, 0);
    let key = CacheInstanceKey {
        key_hi,
        key_lo,
        item_key,
        substs_hash: sh,
        stamps_digest: sd,
        flags: 0,
    };
    let inlined_key = stable_file_key(&outline, 0, 1).expect("file key");
    let inlined_stamp = stamps.file(0, 1).expect("file stamp");
    let mut enc = BundleEncoder::new(&outline, &types, &reached, &state, &stamps);
    enc.add_instance(&key, true, &code, &relocs, &[(inlined_key, inlined_stamp)])
        .expect("add");
    let sections = enc.finish().expect("finish");
    let path = tmp.join("tables.aot");
    write_bundle(&path, CACHE_MAGIC, CACHE_VERSION, cfg, &sections).expect("write");
    let (backing, hdr) = open_bundle(&path, CACHE_MAGIC, CACHE_VERSION, cfg).expect("open");
    let file_index = LocalFileIndex::build(&outline);
    let shared = SharedTables::decode(
        backing.as_slice(),
        &hdr,
        &outline,
        &types,
        &file_index,
        &stamps,
    )
    .expect("decode tables");
    let rec = hdr.records_off;
    let p_off = read_u32_le(backing.as_slice(), rec + 16) as usize;
    let p_len = read_u32_le(backing.as_slice(), rec + 20) as usize;
    let payload = &backing.as_slice()[hdr.blob_off + p_off..hdr.blob_off + p_off + p_len];
    assert_eq!(read_u32_le(payload, 36), 1);
    assert_eq!(read_u64_le(payload, 40), inlined_key);
    assert_eq!(read_u64_le(payload, 48), inlined_stamp);
    let mut cur = Cursor::new(&payload[56..]);
    let dummy = DummySlots {
        rodatas: std::sync::Mutex::new(Vec::new()),
        vtables: std::sync::Mutex::new(Vec::new()),
    };
    let decoded_relocs = decode_symbol_list(
        &shared,
        &types,
        &reached,
        &dummy,
        &mut cur,
        relocs.len(),
        |_| Some(RodataId(1)),
        |_| Some(VtableId(0)),
    )
    .expect("decode syms");
    assert_eq!(decoded_relocs.len(), relocs.len());

    let helper_item = ItemRef {
        krate: 0,
        file: 1,
        item: 0,
    };
    let mut edited_outline = Outline::file(&src, CfgTarget::default());
    let main_orig = Source::parse(src.clone(), "fn main() {}\n".to_owned()).expect("parse main");
    let main_orig_items = scan(&main_orig).expect("scan main");
    let helper_orig =
        Source::parse(helper_path.clone(), "fn helper() {}\n".to_owned()).expect("parse");
    let helper_orig_items = scan(&helper_orig).expect("scan");
    edited_outline.crates[0]
        .files
        .push(Arc::new(crate::outline::tree::File::new(
            main_orig,
            main_orig_items,
            false,
        )));
    edited_outline.crates[0]
        .files
        .push(Arc::new(crate::outline::tree::File::new(
            helper_orig,
            helper_orig_items,
            false,
        )));
    let base_stamps = Stamps::compute(&edited_outline, 0, 1);
    let helper_mod = Source::parse(
        helper_path.clone(),
        "fn helper() { let _x = 1; }\n".to_owned(),
    )
    .expect("parse");
    let helper_mod_items = scan(&helper_mod).expect("scan");
    edited_outline.crates[0].files[1] = Arc::new(crate::outline::tree::File::new(
        helper_mod,
        helper_mod_items,
        false,
    ));
    let body_stamps = Stamps::compute(&edited_outline, 0, 1);
    assert_eq!(body_stamps.iface(0), base_stamps.iface(0));
    assert_eq!(body_stamps.file(0, 0), base_stamps.file(0, 0));
    assert_ne!(body_stamps.file(0, 1), base_stamps.file(0, 1));
    assert_eq!(
        touched_stamps_digest(&types, &body_stamps, item, substs),
        touched_stamps_digest(&types, &base_stamps, item, substs),
    );
    assert_ne!(
        touched_stamps_digest(&types, &body_stamps, helper_item, substs),
        touched_stamps_digest(&types, &base_stamps, helper_item, substs),
    );

    let sig_mod =
        Source::parse(helper_path.clone(), "fn helper2() {}\n".to_owned()).expect("parse");
    let sig_mod_items = scan(&sig_mod).expect("scan");
    edited_outline.crates[0].files[1] = Arc::new(crate::outline::tree::File::new(
        sig_mod,
        sig_mod_items,
        false,
    ));
    let sig_stamps = Stamps::compute(&edited_outline, 0, 1);
    assert_ne!(sig_stamps.iface(0), base_stamps.iface(0));

    let stale_src = Source::parse(helper_path, "fn helper2() {}\n".to_owned()).expect("parse");
    let stale_items = scan(&stale_src).expect("scan");
    outline.crates[0].files[1] = Arc::new(crate::outline::tree::File::new(
        stale_src,
        stale_items,
        false,
    ));
    let stale_stamps = Stamps::compute(&outline, 0, 1);
    let stale_tables = SharedTables::decode(
        backing.as_slice(),
        &hdr,
        &outline,
        &types,
        &file_index,
        &stale_stamps,
    )
    .expect("decode stale");
    assert!(stale_tables.item_at(0).is_none());
    drop(backing);
    drop(std::fs::remove_dir_all(&tmp));
}
