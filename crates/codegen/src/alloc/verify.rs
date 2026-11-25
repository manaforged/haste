use std::fmt::Debug;

use crate::ir::{ArgumentPurpose, Block, Function};
use crate::live::{Live, Range, overlaps};
use crate::park::scratch_clobbers;

use super::{Location, Plan, Regs};

fn crosses_call_strictly(ranges: &[Range], at: u32) -> bool {
    ranges.iter().any(|r| r.start < at && at < r.end)
}

fn verify_call_parked<R: Copy + PartialEq + Debug>(
    live: &Live,
    plan: &Plan<R>,
    ranges: &[Range],
    v: usize,
    reg_label: impl FnOnce() -> String,
) -> Result<(), String> {
    for (&at, &(inst, _)) in live.calls.iter().zip(&live.sites) {
        if crosses_call_strictly(ranges, at)
            && !plan
                .parked_at(inst)
                .iter()
                .any(|(pv, _, _)| pv.index() == v)
        {
            return Err(format!(
                "regalloc call clobber: v{v} in caller-saved {} crosses call {inst:?} unparked",
                reg_label()
            ));
        }
    }
    Ok(())
}

fn copy_roots(count: usize, copies: &[(usize, usize)]) -> Vec<usize> {
    let mut parent: Vec<usize> = (0..count).collect();
    for &(copy, source) in copies {
        if copy < count && source < count {
            parent[copy] = source;
        }
    }
    for v in 0..count {
        let mut root = v;
        let mut steps = 0;
        while parent[root] != root && steps < count {
            root = parent[root];
            steps += 1;
        }
        parent[v] = root;
    }
    parent
}

fn verify_no_overlap<R: Copy + PartialEq + Debug>(
    live: &Live,
    roots: &[usize],
    kind: &str,
    grouped: Vec<(R, Vec<usize>)>,
) -> Result<(), String> {
    for (reg, vals) in grouped {
        for (i, &a) in vals.iter().enumerate() {
            let ra = live.ranges_of(a);
            if ra.is_empty() {
                continue;
            }
            for &b in &vals[i + 1..] {
                if roots[a] == roots[b] {
                    continue;
                }
                let rb = live.ranges_of(b);
                if !rb.is_empty() && overlaps(ra, rb) {
                    return Err(format!(
                        "regalloc conflict: v{a} and v{b} both assigned {kind} {reg:?} with overlapping ranges"
                    ));
                }
            }
        }
    }
    Ok(())
}

fn push_group<R: Copy + PartialEq>(groups: &mut Vec<(R, Vec<usize>)>, reg: R, val: usize) {
    if let Some((_, list)) = groups.iter_mut().find(|(r, _)| *r == reg) {
        list.push(val);
    } else {
        groups.push((reg, vec![val]));
    }
}

/// Verifies register-allocation invariants for `plan` against `live` and `regs`.
pub(crate) fn verify_plan<R: Copy + PartialEq + Debug>(
    func: &Function,
    live: &Live,
    plan: &Plan<R>,
    regs: &Regs<R>,
    (order, copied, copies): (&[Block], &[bool], &[(usize, usize)]),
) -> Result<(), String> {
    let has_dict = func
        .signature
        .params
        .iter()
        .any(|p| p.purpose == ArgumentPurpose::Dict);
    let mut int_groups: Vec<(R, Vec<usize>)> = Vec::new();
    let mut float_groups: Vec<(u8, Vec<usize>)> = Vec::new();
    for (v, loc) in plan.locations.iter().enumerate() {
        if plan.elided[v] || plan.fused[v] {
            continue;
        }
        match *loc {
            Location::Reg(r) => {
                let valid = regs.saved.contains(&r)
                    || regs.leaf.contains(&r)
                    || (!has_dict && regs.dict == Some(r));
                if !valid {
                    return Err(format!(
                        "regalloc invalid int register {r:?} assigned to v{v}"
                    ));
                }
                if regs.saved.contains(&r) && !plan.used.contains(&r) {
                    return Err(format!(
                        "regalloc callee-saved int register {r:?} used by v{v} missing from plan.used"
                    ));
                }
                push_group(&mut int_groups, r, v);
            }
            Location::Float(f) => {
                if !regs.saved_float.contains(&f) && !regs.float.contains(&f) {
                    return Err(format!(
                        "regalloc invalid float register f{f} assigned to v{v}"
                    ));
                }
                if regs.saved_float.contains(&f) && !plan.used_floats.contains(&f) {
                    return Err(format!(
                        "regalloc callee-saved float register f{f} used by v{v} missing from plan.used_floats"
                    ));
                }
                push_group(&mut float_groups, f, v);
            }
            _ => {}
        }
    }
    let roots = copy_roots(plan.locations.len(), copies);
    verify_no_overlap(live, &roots, "int", int_groups)?;
    verify_no_overlap(live, &roots, "float", float_groups)?;

    let preserved_ranges: Vec<Range> = live
        .preserved_calls
        .iter()
        .map(|&at| Range {
            start: at - 1,
            end: at - 1,
        })
        .collect();
    let clobbers = scratch_clobbers(
        func,
        (regs.wide_scratch, regs.op_scratch, regs.scratch_mask),
        (order, copied, &plan.elided),
    );
    for (v, loc) in plan.locations.iter().enumerate() {
        if plan.elided[v] || plan.fused[v] {
            continue;
        }
        let ranges = live.ranges_of(v);
        if ranges.is_empty() {
            continue;
        }
        match *loc {
            Location::Reg(r) => {
                for (clob_regs, clob_ranges) in &clobbers {
                    if clob_regs.contains(&r) && overlaps(ranges, clob_ranges) {
                        return Err(format!(
                            "regalloc clobber: v{v} in {r:?} overlaps scratch clobber range"
                        ));
                    }
                }
                if !regs.saved.contains(&r) {
                    if !preserved_ranges.is_empty()
                        && regs.preserved_clobbers.contains(&r)
                        && overlaps(ranges, &preserved_ranges)
                    {
                        return Err(format!(
                            "regalloc clobber: v{v} in {r:?} overlaps preserved-call clobber"
                        ));
                    }
                    verify_call_parked(live, plan, ranges, v, || format!("{r:?}"))?;
                }
            }
            Location::Float(f) if !regs.saved_float.contains(&f) => {
                if !preserved_ranges.is_empty() && overlaps(ranges, &preserved_ranges) {
                    return Err(format!(
                        "regalloc clobber: v{v} in f{f} overlaps preserved-call clobber"
                    ));
                }
                verify_call_parked(live, plan, ranges, v, || format!("f{f}"))?;
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alloc::{AtomicAddrMode, plan};
    use crate::frame::{Abi, Loc};
    use crate::ir::{AbiParam, CallConv, Op, Signature, Type};

    fn dummy_scratch(_: &Function, _: &Op) -> u8 {
        0
    }

    fn dummy_imm(_: i64) -> u64 {
        1
    }

    const TEST_REGS: Regs<u8> = Regs {
        saved: &[10, 11],
        saved_float: &[],
        leaf: &[0, 1, 2],
        wide_scratch: &[],
        op_scratch: &[],
        preserved_clobbers: &[],
        scratch_mask: dummy_scratch,
        dict: None,
        float: &[0, 1],
        float_branches: &[],
        vectors: false,
        absolute_calls: false,
        constant_calls: false,
        two_address: false,
        park: true,
        fold_sib: false,
        atomic_addrs: AtomicAddrMode::None,
        call_zext: false,
        imm_store: dummy_imm,
        high_product: None,
    };

    fn test_classify(sig: &Signature) -> Result<Abi<u8>, String> {
        Ok(Abi {
            params: (0..sig.params.len()).map(|i| Loc::Int(i as u8)).collect(),
            returns: (0..sig.returns.len()).map(|i| Loc::Int(i as u8)).collect(),
            stack: 0,
        })
    }

    #[test]
    fn verify_plan_catches_overlap_missing_save_and_call_clobber() {
        let mut sig = Signature::new(CallConv::Internal);
        sig.params.push(AbiParam::new(Type::I64));
        sig.params.push(AbiParam::new(Type::I64));
        sig.returns.push(AbiParam::new(Type::I64));
        let mut callee_sig = Signature::new(CallConv::Internal);
        callee_sig.returns.push(AbiParam::new(Type::I64));

        let mut func = Function::new(sig);
        let mut b = crate::FunctionBuilder::new(&mut func);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        let csig = b.import_signature(callee_sig);
        b.switch_to_block(entry);
        let p0 = b.block_params(entry)[0];
        let p1 = b.block_params(entry)[1];
        let sym = b.ins().symbol(1);
        let call = b.ins().call_indirect(csig, sym, &[]);
        let res = b.inst_results(call)[0];
        let s1 = b.ins().iadd(p0, p1);
        let s2 = b.ins().iadd(s1, res);
        b.ins().return_(&[s2]);
        func.entry = Some(entry);

        let order = crate::live::order(&func);
        let copied = vec![false; func.insts.len()];
        let abi = test_classify(&func.signature).expect("classify");
        let valid = plan(
            &func,
            &TEST_REGS,
            (&order, &copied, true, false),
            &abi.params,
            &test_classify,
        )
        .expect("valid plan");
        let elided = crate::live::Elided {
            sib: &valid.sib,
            atomic_base: &[],
            elided: &valid.elided,
        };
        let wanted = vec![true; func.types.len()];
        let live = crate::live::live(&func, &order, &copied, &wanted, elided, &[]);
        assert!(verify_plan(&func, &live, &valid, &TEST_REGS, (&order, &copied, &[])).is_ok());

        let mut bad = valid;
        bad.locations[p0.index()] = Location::Reg(10);
        bad.locations[p1.index()] = Location::Reg(10);
        if !bad.used.contains(&10) {
            bad.used.push(10);
        }
        let err = verify_plan(&func, &live, &bad, &TEST_REGS, (&order, &copied, &[]))
            .expect_err("overlap must fail");
        assert!(err.contains("conflict"), "unexpected error: {err}");

        let saved_used = bad.used.clone();
        bad.locations[p1.index()] = Location::Slot;
        bad.used.clear();
        let err = verify_plan(&func, &live, &bad, &TEST_REGS, (&order, &copied, &[]))
            .expect_err("missing callee-saved entry must fail");
        assert!(err.contains("plan.used"), "unexpected error: {err}");

        bad.used = saved_used;
        bad.locations[p0.index()] = Location::Reg(2);
        bad.parked.clear();
        let err = verify_plan(&func, &live, &bad, &TEST_REGS, (&order, &copied, &[]))
            .expect_err("unparked call-crossing caller-saved reg must fail");
        assert!(err.contains("call clobber"), "unexpected error: {err}");
    }
}
