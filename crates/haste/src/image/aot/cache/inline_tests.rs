use super::*;
use crate::graph::cfg::Target as CfgTarget;
use crate::host::HostSymbol;
use crate::lower::{RodataReloc, Slots, SymbolSource, VtableEntry, Word};
use crate::names::Primitive;
use crate::outline::scan::scan;
use crate::outline::source::Source;
use crate::sema::ty::{Kind, Types};

fn sample_inline_function() -> Arc<haste_codegen::Function> {
    let mut sig = haste_codegen::Signature::new(haste_codegen::CallConv::Internal);
    sig.params
        .push(haste_codegen::AbiParam::new(haste_codegen::Type::I64));
    sig.returns
        .push(haste_codegen::AbiParam::new(haste_codegen::Type::I64));
    let mut func = haste_codegen::Function::new(sig);
    let mut fb = haste_codegen::FunctionBuilder::new(&mut func);
    let entry = fb.create_block();
    fb.append_block_params_for_function_params(entry);
    fb.switch_to_block(entry);
    let arg = fb.block_params(entry)[0];
    let one = fb.ins().iconst(haste_codegen::Type::I64, 1);
    let sum = fb.ins().iadd(arg, one);
    fb.ins().return_(&[sum]);
    Arc::new(func)
}

#[test]
fn reloc_plan_encode_decode_round_trip() {
    let tmp = std::env::temp_dir().join(format!("haste-plan-test-{}", std::process::id()));
    drop(std::fs::remove_dir_all(&tmp));
    std::fs::create_dir_all(&tmp).expect("mkdir");
    let src = tmp.join("main.rs");
    std::fs::write(&src, "fn main() {}\n").expect("write");
    let mut outline = Outline::file(&src, CfgTarget::default());
    let main_src = Source::parse(src.clone(), "fn main() {}\n".to_owned()).expect("parse main");
    let main_items = scan(&main_src).expect("scan main");
    outline.crates[0]
        .files
        .push(Arc::new(crate::outline::tree::File::new(
            main_src, main_items, false,
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
    let ro = state.alloc_rodata(b"hello", &[]);
    let vt = state.intern_vtable(&[VtableEntry::Word(Word::Code(0))]);

    let sub_plan = Arc::new(RelocPlan::from_vec(vec![
        crate::lower::reloc::RelocStep::Const(SymbolSource::Word(77)),
        crate::lower::reloc::RelocStep::SiteSlot(crate::lower::reloc::SiteOrigin::Known(
            item, substs,
        )),
    ]));
    let plan = RelocPlan::from_vec(vec![
        crate::lower::reloc::RelocStep::Const(SymbolSource::Slot(0)),
        crate::lower::reloc::RelocStep::Const(SymbolSource::Rodata(ro)),
        crate::lower::reloc::RelocStep::Const(SymbolSource::Vtable(vt)),
        crate::lower::reloc::RelocStep::Const(SymbolSource::Host(HostSymbol::Alloc)),
        crate::lower::reloc::RelocStep::Promoted { seq: 3, size: 16 },
        crate::lower::reloc::RelocStep::VtableAt(crate::body::ExprId(5)),
        crate::lower::reloc::RelocStep::VtableSlot {
            expr: crate::body::ExprId(6),
            slot: 2,
        },
        crate::lower::reloc::RelocStep::SiteInline {
            origin: crate::lower::reloc::SiteOrigin::Known(item, substs),
            plan: Arc::clone(&sub_plan),
        },
        crate::lower::reloc::RelocStep::SiteInline {
            origin: crate::lower::reloc::SiteOrigin::ShapeGlueDrop,
            plan: sub_plan,
        },
        crate::lower::reloc::RelocStep::TypeId(crate::lower::reloc::TyOrigin::Subst(0)),
        crate::lower::reloc::RelocStep::TypeName(crate::lower::reloc::TyOrigin::Local(1)),
    ]);

    let mut enc = BundleEncoder::new(&outline, &types, &reached, &state, &stamps);
    let encoded_plan = plan::encode_reloc_plan(&mut enc, &plan).expect("encode plan");
    enc.add_raw_record((1, 2), &encoded_plan).expect("record");
    let sections = enc.finish().expect("finish");
    let cfg = tests::sample_config(0, 1, 2);
    let path = tmp.join("plan.inl");
    write_bundle(
        &path,
        inline::INL_MAGIC,
        inline::INL_VERSION,
        cfg,
        &sections,
    )
    .expect("write");

    let view = BundleView::open(&path, inline::INL_MAGIC, inline::INL_VERSION, cfg);
    let file_index = OnceLock::new();
    let tables = view
        .shared_tables(&outline, &types, &file_index, &stamps, None)
        .expect("tables");
    let backing = view.backing.as_ref().expect("backing").as_slice();
    let rec = view.header.records_off;
    let off = read_u32_le(backing, rec + 16);
    let len = read_u32_le(backing, rec + 20);
    let blob = view.blob_slice(off, len).expect("blob");
    let ro_out = Mutex::new(Vec::new());
    let vt_out = Mutex::new(Vec::new());
    struct PlanSlots<'a> {
        ro: &'a Mutex<Vec<(Vec<u8>, Vec<RodataReloc>)>>,
        vt: &'a Mutex<Vec<Vec<VtableEntry>>>,
        bodies: crate::lower::Bodies,
    }
    impl Slots for PlanSlots<'_> {
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
        fn bodies(&self) -> &crate::lower::Bodies {
            &self.bodies
        }
        fn rodata(&self, bytes: &[u8], relocs: &[RodataReloc]) -> crate::lower::RodataId {
            let mut g = lock(self.ro);
            let id = crate::lower::RodataId(g.len() as u32);
            g.push((bytes.to_vec(), relocs.to_vec()));
            id
        }
        fn vtable_id(&self, entries: &[VtableEntry]) -> crate::lower::VtableId {
            let mut g = lock(self.vt);
            let id = crate::lower::VtableId(g.len() as u32);
            g.push(entries.to_vec());
            id
        }
    }
    let test_slots = PlanSlots {
        ro: &ro_out,
        vt: &vt_out,
        bodies: Default::default(),
    };
    let decoded = plan::decode_reloc_plan(
        tables,
        &types,
        &reached,
        &test_slots,
        blob,
        |id| view.decode_rodata(tables, &test_slots, &types, &reached, id),
        |id| view.decode_vtable(tables, &test_slots, &types, &reached, id),
    )
    .expect("decode plan");
    assert_eq!(decoded.steps.len(), plan.steps.len());
    assert_eq!(decoded.reloc_len(), plan.reloc_len());
    assert_eq!(decoded.has_calls(), plan.has_calls());
    assert_eq!(decoded.has_fn_ptrs(), plan.has_fn_ptrs());
    drop(view);
    drop(std::fs::remove_dir_all(&tmp));
}

#[test]
fn inline_template_bundle_round_trip_and_invalidation() {
    let tmp = std::env::temp_dir().join(format!("haste-inl-cache-test-{}", std::process::id()));
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
    let names = crate::names::Names::default();
    let cx = Context::new(&outline, &names, crate::sema::Cache::default());
    let u64_ty = cx.types.intern(Kind::Prim(Primitive::U64));
    let substs = cx.types.intern_substs(&[u64_ty]);
    let item = ItemRef {
        krate: 0,
        file: 0,
        item: 0,
    };
    let helper_item = ItemRef {
        krate: 0,
        file: 1,
        item: 0,
    };
    let mut reached = Reached::default();
    reached.list.push((
        (item, substs),
        Ok(Arc::new(crate::sema::infer::Typed::default())),
    ));
    reached.list.push((
        (helper_item, crate::sema::substs::Substs::EMPTY),
        Ok(Arc::new(crate::sema::infer::Typed::default())),
    ));
    reached.index.insert((item, substs), 0);
    reached
        .index
        .insert((helper_item, crate::sema::substs::Substs::EMPTY), 1);

    let stamps = Stamps::compute(&outline, 0, 1);
    let file_index = OnceLock::new();
    let cfg = tests::sample_config(0b10_0010, 1, 2);
    let path = tmp.join("bundle.inl");
    let mut local_stamps = HashMap::default();
    let main_fkey = stable_file_key(&outline, 0, 0).expect("main fkey");
    let helper_fkey = stable_file_key(&outline, 0, 1).expect("helper fkey");
    local_stamps.insert(main_fkey, stamps.file(0, 0).expect("main stamp"));
    local_stamps.insert(helper_fkey, stamps.file(0, 1).expect("helper stamp"));

    let func = sample_inline_function();
    let plan = Arc::new(RelocPlan::from_vec(vec![
        crate::lower::reloc::RelocStep::SiteInline {
            origin: crate::lower::reloc::SiteOrigin::Known(
                helper_item,
                crate::sema::substs::Substs::EMPTY,
            ),
            plan: Arc::new(RelocPlan::from_vec(vec![
                crate::lower::reloc::RelocStep::Const(SymbolSource::Word(99)),
            ])),
        },
    ]));
    let template = Arc::new(InlineTemplate {
        body: Arc::clone(&func),
        returns: true,
        reloc_plan: Arc::clone(&plan),
        height: 2,
        size: 3,
        form: crate::lower::stencil::Form::Compact,
    });
    let relocs = vec![SymbolSource::Word(99)];
    let pos_fp = FootprintId(0xDEAD_BEEF_u128);
    let neg_fp = FootprintId(0xCAFE_BABE_u128);
    let pos_parts = (0x1111_u64, 0x2222_u64, 0x3333_u64);
    let neg_parts = (0x4444_u64, 0x5555_u64, 0x6666_u64);

    let state = AotState::new(0).into_inner().expect("lock");
    let inl = InlCache::open(path.clone(), cfg);
    inl.record(
        (&cx, &reached),
        pos_parts,
        cfg.config_digest,
        (0, 80, true, false, false, false, true),
        pos_fp,
        Some(&template),
        &plan,
        &relocs,
    );
    inl.record(
        (&cx, &reached),
        neg_parts,
        cfg.config_digest,
        (1, 80, true, false, false, false, false),
        neg_fp,
        None,
        &RelocPlan::default(),
        &[],
    );
    let (flushed, _) = inl.flush(
        &state,
        (&cx, &reached),
        cfg,
        &file_index,
        &stamps,
        &local_stamps,
    );
    assert_eq!(flushed, 2);

    struct EmptySlots(crate::lower::Bodies);
    impl Slots for EmptySlots {
        fn entry(&self, _: usize) -> i64 {
            0
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
        fn bodies(&self) -> &crate::lower::Bodies {
            &self.0
        }
    }
    let slots = EmptySlots(Default::default());
    let reopened = InlCache::open(path.clone(), cfg);
    let pos_key_shape = reopened
        .key_for_entry(
            pos_parts,
            cfg.config_digest,
            (80, true, false, false, false, true),
        )
        .expect("pos key");
    match reopened.lookup(
        (&slots, &cx, &reached),
        pos_key_shape,
        &file_index,
        &stamps,
        &local_stamps,
        true,
    ) {
        Some(InlineCacheHit::Shape(fp, returns, height, size)) => {
            assert_eq!(fp, pos_fp);
            assert!(returns);
            assert_eq!(height, 2);
            assert_eq!(size, 3);
        }
        _ => panic!("expected shape hit"),
    }

    let pos_key_body = reopened
        .key_for_entry(
            pos_parts,
            cfg.config_digest,
            (80, true, false, false, false, true),
        )
        .expect("pos key body");
    match reopened.lookup(
        (&slots, &cx, &reached),
        pos_key_body,
        &file_index,
        &stamps,
        &local_stamps,
        false,
    ) {
        Some(InlineCacheHit::Body(fp, decoded)) => {
            assert_eq!(fp, pos_fp);
            assert_eq!(decoded.body.encode(), func.encode());
            assert!(decoded.returns);
            assert_eq!(decoded.height, 2);
            assert_eq!(decoded.size, 3);
            assert_eq!(decoded.form as u8, template.form as u8);
            assert_eq!(decoded.reloc_plan.reloc_len(), 1);
        }
        _ => panic!("expected body hit"),
    }

    let neg_key = reopened
        .key_for_entry(
            neg_parts,
            cfg.config_digest,
            (80, true, false, false, false, false),
        )
        .expect("neg key");
    assert!(matches!(
        reopened.lookup(
            (&slots, &cx, &reached),
            neg_key,
            &file_index,
            &stamps,
            &local_stamps,
            false,
        ),
        Some(InlineCacheHit::Negative(fp)) if fp == neg_fp
    ));

    let bad_limit = reopened
        .key_for_entry(
            pos_parts,
            cfg.config_digest,
            (81, true, false, false, false, true),
        )
        .expect("bad limit");
    assert!(
        reopened
            .lookup(
                (&slots, &cx, &reached),
                bad_limit,
                &file_index,
                &stamps,
                &local_stamps,
                false,
            )
            .is_none()
    );

    let mut stale_locals = local_stamps.clone();
    stale_locals.insert(helper_fkey, 999_999);
    let stale_key = reopened
        .key_for_entry(
            pos_parts,
            cfg.config_digest,
            (80, true, false, false, false, true),
        )
        .expect("stale key");
    assert!(
        reopened
            .lookup(
                (&slots, &cx, &reached),
                stale_key,
                &file_index,
                &stamps,
                &stale_locals,
                false,
            )
            .is_none()
    );

    drop(reopened);
    drop(std::fs::remove_dir_all(&tmp));
}
