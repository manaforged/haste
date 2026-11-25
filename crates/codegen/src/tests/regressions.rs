use super::{build_fn, full_arm64, full_x64};
use crate::ir::{Convert, Op, Type};
use crate::promote::Moves;
use crate::{AbiParam, CallConv, Effort, IntCC, Signature, Target, compile, schedule, verify};

#[test]
fn verify_rejects_illegal_convert_widths() {
    let mut sig = Signature::new(CallConv::Internal);
    sig.params.push(AbiParam::new(Type::I64));
    sig.returns.push(AbiParam::new(Type::I128));
    let bad_reduce = build_fn(sig, |b, p| {
        let wide = b.ins().ireduce(Type::I128, p[0]);
        b.ins().return_(&[wide]);
    });
    assert!(
        verify::verify(&bad_reduce, &full_x64()).is_err(),
        "Ireduce from I64 to I128 must fail IR verification"
    );

    let mut sig2 = Signature::new(CallConv::Internal);
    sig2.params.push(AbiParam::new(Type::I64));
    sig2.returns.push(AbiParam::new(Type::I32));
    let bad_extend = build_fn(sig2, |b, p| {
        let narrow = b.ins().uextend(Type::I32, p[0]);
        b.ins().return_(&[narrow]);
    });
    assert!(
        verify::verify(&bad_extend, &full_x64()).is_err(),
        "Uextend from I64 to I32 must fail IR verification"
    );
}

#[test]
fn widen_preserves_uextend_when_target_is_i128() {
    let mut sig = Signature::new(CallConv::Internal);
    sig.params.push(AbiParam::new(Type::I8));
    sig.params.push(AbiParam::new(Type::I32));
    sig.returns.push(AbiParam::new(Type::I128));
    let mut func = build_fn(sig, |b, p| {
        let head = b.create_block();
        let exit = b.create_block();
        let acc = b.append_block_param(head, Type::I8);
        let cnt = b.append_block_param(head, Type::I32);
        let zero = b.ins().iconst(Type::I32, 0);
        b.ins().jump(head, &[p[0], zero]);

        b.switch_to_block(head);
        let a32 = b.ins().uextend(Type::I32, acc);
        let a64 = b.ins().uextend(Type::I64, acc);
        let next = b.ins().iadd_imm(a32, 1);
        let next8 = b.ins().ireduce(Type::I8, next);
        let next_cnt = b.ins().iadd_imm(cnt, 1);
        let keep = b.ins().icmp(IntCC::UnsignedLessThan, next_cnt, p[1]);
        let _ = a64;
        b.ins().brif(keep, head, &[next8, next_cnt], exit, &[]);

        b.switch_to_block(exit);
        let wide = b.ins().uextend(Type::I128, acc);
        b.ins().return_(&[wide]);
    });
    let target = full_x64();
    schedule::run(&mut func, Moves::Split, &target);
    verify::verify(&func, &target).expect("widened I8->I128 stays valid IR");
    for inst in &func.insts {
        if let Op::Convert(Convert::Ireduce, src) = inst.op
            && let Some(res) = inst.result
        {
            assert!(
                func.ty(res).bits() < func.ty(src).bits(),
                "Ireduce must narrow: {:?} -> {:?}",
                func.ty(src),
                func.ty(res)
            );
        }
    }
}

#[test]
fn pairs_split_skips_empty_unreachable_blocks() {
    let mut sig = Signature::new(CallConv::Internal);
    sig.params.push(AbiParam::new(Type::I128));
    sig.returns.push(AbiParam::new(Type::I128));
    let mut func = build_fn(sig, |b, p| {
        let dead = b.create_block();
        let _dead_param = b.append_block_param(dead, Type::I128);
        let one = b.ins().iconst(Type::I64, 1);
        let zero = b.ins().iconst(Type::I64, 0);
        let one128 = b.ins().iconcat(one, zero);
        let sum = b.ins().iadd(p[0], one128);
        b.ins().return_(&[sum]);
    });
    let target = full_x64();
    assert!(verify::verify(&func, &target).is_ok());
    assert!(crate::pairs::split(&mut func));
    verify::verify(&func, &target)
        .expect("empty unreachable block remains valid after pairs::split");
}

#[test]
fn sink_frontier_preserves_store_on_live_self_loop_backedge() {
    let mut sig = Signature::new(CallConv::Internal);
    sig.params.push(AbiParam::new(Type::I64));
    sig.returns.push(AbiParam::new(Type::I64));
    let mut func = build_fn(sig, |b, p| {
        let slot = b.create_stack_slot(8, 3);
        let loop_b = b.create_block();
        let after = b.create_block();
        let trap_b = b.create_block();
        let done = b.create_block();
        let i = b.append_block_param(loop_b, Type::I64);
        let zero = b.ins().iconst(Type::I64, 0);
        b.ins().stack_store(zero, slot, 0);
        b.ins().jump(loop_b, &[zero]);

        b.switch_to_block(loop_b);
        let cur = b.ins().stack_load(Type::I64, slot, 0);
        let next_val = b.ins().iadd_imm(cur, 7);
        b.ins().stack_store(next_val, slot, 0);
        let next_i = b.ins().iadd_imm(i, 1);
        let cont = b.ins().icmp(IntCC::SignedLessThan, next_i, p[0]);
        b.ins().brif(cont, loop_b, &[next_i], after, &[]);

        b.switch_to_block(after);
        let is_neg = b.ins().icmp_imm(IntCC::SignedLessThan, p[0], 0);
        b.ins().brif(is_neg, trap_b, &[], done, &[]);

        b.switch_to_block(trap_b);
        b.ins().trap();

        b.switch_to_block(done);
        let addr = b.ins().stack_addr(Type::I64, slot, 0);
        let out = b.ins().load(Type::I64, addr, 0);
        b.ins().return_(&[out]);
    });
    let target = full_x64();
    let _ = crate::loads::run(&mut func, false, &target);
    let loop_stores = func.blocks[1]
        .insts
        .iter()
        .filter(|&&inst| matches!(func.insts[inst.index()].op, Op::StackStore(..)))
        .count();
    assert!(
        loop_stores > 0,
        "StackStore in self-loop block with live-in read must not be sunk out of the loop"
    );
}

fn arm64_words(code: crate::Code) -> Vec<u32> {
    code.bytes()
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn arm64_shl32_const_uses_32bit_zero_extending_instruction() {
    let mut sig = Signature::new(CallConv::Internal);
    sig.params.push(AbiParam::new(Type::I32));
    sig.returns.push(AbiParam::new(Type::I64));
    let func = build_fn(sig, |b, p| {
        let sh = b.ins().iconst(Type::I32, 4);
        let shifted = b.ins().ishl(p[0], sh);
        let wide = b.ins().uextend(Type::I64, shifted);
        b.ins().return_(&[wide]);
    });
    let mut quick_arm64 = *Target::find("aarch64-apple-darwin").expect("arm64 darwin target");
    quick_arm64.effort = Effort::Quick;
    for target in [quick_arm64, full_arm64()] {
        let words = arm64_words(compile(func.clone(), &target).expect("arm64 shl32 compiles"));
        let has_32bit_ubfm = words.iter().any(|&w| (w & 0xFFC0_0000) == 0x5300_0000);
        let has_ubfiz28 = words.iter().any(|&w| (w & 0xFFFF_FC00) == 0xD37C_6C00);
        let has_64bit_lsl4 = words.iter().any(|&w| (w & 0xFFFF_FC00) == 0xD37C_EC00);
        assert!(
            (has_32bit_ubfm || has_ubfiz28) && !has_64bit_lsl4,
            "expected zero-extending shift (32-bit UBFM or UBFIZ #4,#28) and no 64-bit LSL X #4 in {words:#x?}"
        );
    }
}

#[test]
fn arm64_extr_preserves_uextend_when_kr_exceeds_inner_width() {
    let mut sig = Signature::new(CallConv::Internal);
    sig.params.push(AbiParam::new(Type::I16));
    sig.params.push(AbiParam::new(Type::I64));
    sig.returns.push(AbiParam::new(Type::I64));
    let func = build_fn(sig, |b, p| {
        let neg = b.ins().ineg(p[0]);
        let wide = b.ins().uextend(Type::I64, neg);
        let c4 = b.ins().iconst(Type::I64, 4);
        let c60 = b.ins().iconst(Type::I64, 60);
        let hi = b.ins().ishl(wide, c4);
        let lo = b.ins().ushr(p[1], c60);
        let out = b.ins().bor(hi, lo);
        b.ins().return_(&[out]);
    });
    let words =
        arm64_words(crate::arm64::compile(&func, &full_arm64()).expect("arm64 extr compiles"));
    let has_uext16_or_and16 = words.iter().any(|&w| {
        (w & 0xFFFF_FC00) == 0x5300_3C00
            || (w & 0xFFFF_FC00) == 0xD340_3C00
            || (w & 0xFFC0_0000) == 0x9240_0000
    });
    let has_extr60 = words.iter().any(|&w| (w & 0xFFE0_FC00) == 0x93C0_F000);
    assert!(
        has_uext16_or_and16 || !has_extr60,
        "EXTR #60 must not consume an unextended dirty I16: {words:#x?}"
    );
}

#[test]
fn arm64_select_then_brif_uses_live_flags_instead_of_cbz() {
    let mut sig = Signature::new(CallConv::Internal);
    sig.params.push(AbiParam::new(Type::I64));
    sig.params.push(AbiParam::new(Type::I64));
    sig.params.push(AbiParam::new(Type::I64));
    sig.params.push(AbiParam::new(Type::I64));
    sig.returns.push(AbiParam::new(Type::I64));
    let func = build_fn(sig, |b, p| {
        let then_b = b.create_block();
        let else_b = b.create_block();
        let sum = b.ins().iadd(p[0], p[1]);
        let zero = b.ins().iconst(Type::I64, 0);
        let eq = b.ins().icmp(IntCC::Equal, sum, zero);
        let picked = b.ins().select(eq, p[1], p[2]);
        b.ins().store(picked, p[3], 0);
        let next = b.ins().iadd(p[1], p[2]);
        b.ins().brif(eq, then_b, &[], else_b, &[]);

        b.switch_to_block(then_b);
        b.ins().return_(&[next]);

        b.switch_to_block(else_b);
        let other = b.ins().iadd_imm(next, 1);
        b.ins().return_(&[other]);
    });
    let words = arm64_words(
        crate::arm64::compile(&func, &full_arm64()).expect("arm64 select+brif compiles"),
    );
    let has_bcond = words.iter().any(|&w| (w & 0xFF00_0010) == 0x5400_0000);
    let has_cbz = words.iter().any(|&w| (w & 0x7E00_0000) == 0x3400_0000);
    assert!(
        has_bcond && !has_cbz,
        "Select+Brif with live flags must emit B.cond, not CBZ/CBNZ: {words:#x?}"
    );
}

#[test]
fn arm64_cmp_imm_i16_signed_cmn_sign_extends_zext_sub() {
    let mut sig = Signature::new(CallConv::Internal);
    sig.params.push(AbiParam::new(Type::I16));
    sig.returns.push(AbiParam::new(Type::I8));
    let func = build_fn(sig, |b, p| {
        let mask = b.ins().iconst(Type::I16, 0xFFFF);
        let clean = b.ins().band(p[0], mask);
        let one = b.ins().iconst(Type::I16, 1);
        let sub = b.ins().isub(clean, one);
        let imm = b.ins().iconst(Type::I16, -2);
        let eq = b.ins().icmp(IntCC::Equal, sub, imm);
        b.ins().return_(&[eq]);
    });
    let words =
        arm64_words(crate::arm64::compile(&func, &full_arm64()).expect("arm64 cmp_imm compiles"));
    let has_cmn2 = words.iter().any(|&w| (w & 0xFFFF_FC1F) == 0x3100_081F);
    let has_sxth = words
        .iter()
        .any(|&w| (w & 0xFFFF_FC00) == 0x1300_3C00 || (w & 0xFFFF_FC00) == 0x9340_3C00);
    assert!(
        has_cmn2 && has_sxth,
        "CMN W, #2 on I16 must sign-extend operand with SXTH: {words:#x?}"
    );
}

#[test]
fn arm64_zext_sub_cmp_does_not_poison_last_cmp_for_equal() {
    let mut sig = Signature::new(CallConv::Internal);
    sig.params.push(AbiParam::new(Type::I8));
    sig.returns.push(AbiParam::new(Type::I8));
    let func = build_fn(sig, |b, p| {
        let mask = b.ins().iconst(Type::I8, 0xFF);
        let clean = b.ins().band(p[0], mask);
        let one = b.ins().iconst(Type::I8, 1);
        let sub = b.ins().isub(clean, one);
        let c255 = b.ins().iconst(Type::I8, 255);
        let lt = b.ins().icmp(IntCC::UnsignedLessThan, sub, c255);
        let eq = b.ins().icmp(IntCC::Equal, sub, c255);
        let both = b.ins().iadd(lt, eq);
        b.ins().return_(&[both]);
    });
    let words =
        arm64_words(crate::arm64::compile(&func, &full_arm64()).expect("arm64 last_cmp compiles"));
    let has_uxtb = words
        .iter()
        .any(|&w| (w & 0xFFFF_FC00) == 0xD340_1C00 || (w & 0xFFFF_FC00) == 0x5300_1C00);
    let cmp_255_count = words
        .iter()
        .filter(|&&w| (w & 0xFFFF_FC1F) == 0x7103_FC1F)
        .count();
    assert!(
        has_uxtb && cmp_255_count >= 2,
        "Equal after UnsignedLessThan on wrap_floor must re-compare with UXTB: {words:#x?}"
    );
}
