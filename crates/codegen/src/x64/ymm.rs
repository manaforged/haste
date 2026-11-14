use super::asm::{Alu, Mem, Reg, Rm, Vl, VexOp, Xmm};
use super::features::Features;
use super::float::{X0, X1, X2};
use super::vector::{float_vec_opcode, is_commutative, shift_count_opcode, vec_alu_opcode, vec_prefix};
use super::vector_narrow::{SLL, SRA, SRL};
use super::Emit;
use crate::frame::{ArgStep, Ext};
use crate::ir::{AbiParam, Binary, Lane, Op, ShuffleMask, TargetOpId, Type, Unary, Value};

const MAP_0F: u8 = 1;
const MAP_0F38: u8 = 2;
const MAP_0F3A: u8 = 3;
const PP_NONE: u8 = 0;
const PP_66: u8 = 1;
const PP_F3: u8 = 2;

const MOVAPS: VexOp = VexOp::new(MAP_0F, PP_NONE, 0x28);
const MOVDQU_LOAD: VexOp = VexOp::new(MAP_0F, PP_F3, 0x6F);
const MOVDQU_STORE: VexOp = VexOp::new(MAP_0F, PP_F3, 0x7F);
const PXOR: VexOp = VexOp::new(MAP_0F, PP_66, 0xEF);
const PCMPEQD: VexOp = VexOp::new(MAP_0F, PP_66, 0x76);
const PUNPCKLQDQ: VexOp = VexOp::new(MAP_0F, PP_66, 0x6C);
const PUNPCKHQDQ: VexOp = VexOp::new(MAP_0F, PP_66, 0x6D);
const PSHUFB: VexOp = VexOp::new(MAP_0F38, PP_66, 0x00);
const PCMPEQB: VexOp = VexOp::new(MAP_0F, PP_66, 0x74);
const PCMPGTB: VexOp = VexOp::new(MAP_0F, PP_66, 0x64);
const PMOVMSKB: VexOp = VexOp::new(MAP_0F, PP_66, 0xD7);
const PSLLVD: VexOp = VexOp::new(MAP_0F38, PP_66, 0x47);
const PSLLVQ: VexOp = VexOp::wide(MAP_0F38, PP_66, 0x47);
const PSRLVD: VexOp = VexOp::new(MAP_0F38, PP_66, 0x45);
const PSRLVQ: VexOp = VexOp::wide(MAP_0F38, PP_66, 0x45);
const PSRAVD: VexOp = VexOp::new(MAP_0F38, PP_66, 0x46);
const PERMQ: VexOp = VexOp::wide(MAP_0F3A, PP_66, 0x00);
const PBLENDD: VexOp = VexOp::new(MAP_0F3A, PP_66, 0x02);
const PALIGNR: VexOp = VexOp::new(MAP_0F3A, PP_66, 0x0F);
const INSERTI128: VexOp = VexOp::new(MAP_0F3A, PP_66, 0x38);
const EXTRACTI128: VexOp = VexOp::new(MAP_0F3A, PP_66, 0x39);
const PERM2I128: VexOp = VexOp::new(MAP_0F3A, PP_66, 0x46);
const VZEROUPPER: [u8; 3] = [0xC5, 0xF8, 0x77];
const HIGH_HALF: u8 = 1;
const LOW_DWORDS: u8 = 0x0F;
const QUADS: usize = 4;
const WHOLE_XMM: Type = Type::Vec { lane: Lane::I64, count: 2 };
const WHOLE_YMM: Type = Type::Vec { lane: Lane::I64, count: 4 };

struct Paired {
    quads: [u8; QUADS],
    op: VexOp,
    imm: Option<u8>,
    swap: bool,
}

const PAIRED: [Paired; 6] = [
    Paired { quads: [0, 4, 2, 6], op: PUNPCKLQDQ, imm: None, swap: false },
    Paired { quads: [4, 0, 6, 2], op: PUNPCKLQDQ, imm: None, swap: true },
    Paired { quads: [1, 5, 3, 7], op: PUNPCKHQDQ, imm: None, swap: false },
    Paired { quads: [5, 1, 7, 3], op: PUNPCKHQDQ, imm: None, swap: true },
    Paired { quads: [5, 0, 7, 2], op: PALIGNR, imm: Some(8), swap: false },
    Paired { quads: [1, 4, 3, 6], op: PALIGNR, imm: Some(8), swap: true },
];

/// Whether the CPU this program runs on reports AVX2.
pub(crate) fn host_avx2() -> bool {
    Features::host().avx2
}

fn broadcast(lane: Lane) -> VexOp {
    let op = match lane {
        Lane::I8 => 0x78,
        Lane::I16 => 0x79,
        Lane::I32 | Lane::F32 => 0x58,
        Lane::I64 | Lane::F64 => 0x59,
    };
    VexOp::new(MAP_0F38, PP_66, op)
}

fn lane_shift(op: Binary, lane: Lane) -> Option<VexOp> {
    match (op, lane) {
        (Binary::Ishl, Lane::I32) => Some(PSLLVD),
        (Binary::Ishl, Lane::I64) => Some(PSLLVQ),
        (Binary::Ushr, Lane::I32) => Some(PSRLVD),
        (Binary::Ushr, Lane::I64) => Some(PSRLVQ),
        (Binary::Sshr, Lane::I32) => Some(PSRAVD),
        _ => None,
    }
}

fn byte_op(id: TargetOpId) -> Option<VexOp> {
    match id {
        TargetOpId::X64Pshufb256 => Some(PSHUFB),
        TargetOpId::X64Pcmpeqb256 => Some(PCMPEQB),
        TargetOpId::X64Pcmpgtb256 => Some(PCMPGTB),
        _ => None,
    }
}

fn quad_imm(quads: [u8; QUADS]) -> u8 {
    quads.iter().enumerate().fold(0, |imm, (at, &quad)| imm | ((quad & 3) << (2 * at)))
}

fn blend_imm(quads: [u8; QUADS]) -> u8 {
    quads.iter().enumerate().filter(|&(_, &quad)| usize::from(quad) >= QUADS).fold(0, |imm, (at, _)| imm | (0b11 << (2 * at)))
}

impl Emit<'_> {
    /// Copies the whole register `src` of a value of type `ty` to `dst`.
    pub(super) fn fmov(&mut self, ty: Type, dst: Xmm, src: Xmm) {
        if ty.is_wide_vec() {
            self.asm.vex(Vl::Y, MOVAPS, (dst, Xmm(0)), src);
        } else {
            self.asm.movaps(dst, src);
        }
    }

    /// Loads the 32 bytes at `from` into `ymm`, which needs no alignment.
    pub(super) fn ymm_load(&mut self, ymm: Xmm, from: Mem) {
        self.asm.vex(Vl::Y, MOVDQU_LOAD, (ymm, Xmm(0)), from);
    }

    /// Stores the 32 bytes of `ymm` to `to`, which needs no alignment.
    pub(super) fn ymm_store(&mut self, to: Mem, ymm: Xmm) {
        self.asm.vex(Vl::Y, MOVDQU_STORE, (ymm, Xmm(0)), to);
    }

    /// Clears the high half of every vector register before control leaves a function that holds 32-byte vectors, unless a value in `crossing` leaves in a whole register.
    pub(super) fn leave_wide(&mut self, crossing: &[AbiParam]) {
        if self.frame.wide && !crossing.iter().any(|param| param.value_type.is_wide_vec()) {
            self.asm.bytes(&VZEROUPPER);
        }
    }

    /// Runs one step that moves a whole vector register to where a call or a return takes it, saving through the stack; a function that holds 32-byte vectors moves all 32 bytes.
    pub(super) fn float_step(&mut self, step: ArgStep<u8>) {
        let whole = if self.frame.wide { WHOLE_YMM } else { WHOLE_XMM };
        let top = Mem::new(Reg::Rsp, 0);
        match step {
            ArgStep::Move(src, dst) => self.fmov(whole, Xmm(dst), Xmm(src)),
            ArgStep::Save(src) => {
                self.asm.alu_imm(Alu::Sub, Reg::Rsp, whole.bytes() as i32);
                self.xmm_store(whole, top, Xmm(src));
            }
            ArgStep::Restore(dst) => {
                self.xmm_load(whole, Xmm(dst), top);
                self.asm.alu_imm(Alu::Add, Reg::Rsp, whole.bytes() as i32);
            }
            ArgStep::Load(value, dst) => self.fget(Xmm(dst), value),
        }
    }

    fn wide_rm(&mut self, value: Value, scratch: Xmm) -> Rm {
        match self.elided_mem(value) {
            Some(mem) => mem.into(),
            None => self.foperand(value, scratch),
        }
    }

    /// Emits the lanewise `op` of two 32-byte vectors.
    pub(super) fn ymm_binary(&mut self, op: Binary, (x, y): (Value, Value), result: Value) -> Result<(), String> {
        let Some((lane, count)) = self.ty(x).vec_shape() else {
            return Err(format!("non-vector type {:?} in ymm_binary", self.ty(x)));
        };
        let dst = self.fdst(result);
        let packed = match (vec_alu_opcode(op, lane), float_vec_opcode(op)) {
            (Some(opcode), _) if lane.is_int() || matches!(op, Binary::Band | Binary::Bor | Binary::Bxor) => Some(VexOp::new(MAP_0F, PP_66, opcode)),
            (_, Some(opcode)) if lane.is_float() => Some(VexOp::new(MAP_0F, if vec_prefix(lane).is_empty() { PP_NONE } else { PP_66 }, opcode)),
            _ => None,
        };
        if let Some(packed) = packed {
            let (x, y) = if is_commutative(op) && self.xmm_of(x).is_none() && self.xmm_of(y).is_some() { (y, x) } else { (x, y) };
            let sx = self.fin(x, X0);
            let sy = self.wide_rm(y, X1);
            self.asm.vex(Vl::Y, packed, (dst, sx), sy);
        } else if matches!(op, Binary::Ishl | Binary::Ushr | Binary::Sshr) && lane.is_int() {
            self.ymm_shift(op, (lane, count), (x, y), dst)?;
        } else {
            return Err(format!("{op:?} of 256-bit vector {lane:?}"));
        }
        self.clobber_xmms(&[X0, X1]);
        self.fput(result, dst);
        Ok(())
    }

    fn ymm_shift(&mut self, op: Binary, (lane, count): (Lane, u8), (x, y): (Value, Value), dst: Xmm) -> Result<(), String> {
        let uniform = matches!(lane, Lane::I16 | Lane::I32) || (lane == Lane::I64 && op != Binary::Sshr);
        if uniform && let Some(scalar) = self.splat_source(y, count) {
            let sx = self.fin(x, X0);
            if let Some(imm) = self.const_of(scalar) {
                let ext = match op {
                    Binary::Ushr => SRL,
                    Binary::Sshr => SRA,
                    _ => SLL,
                };
                self.asm.vex_pshift_imm(Vl::Y, ext, lane.bytes(), dst, sx, ((imm as u32) & (lane.bits() - 1)) as u8);
                return Ok(());
            }
            self.get(Reg::Rax, scalar, Ext::Zero);
            self.asm.alu_imm32(Alu::And, Reg::Rax, (lane.bits() - 1) as i32);
            self.clobber_reg(Reg::Rax);
            self.emit_mov_xmm_gpr(X1, Reg::Rax, false);
            self.asm.vex(Vl::Y, VexOp::new(MAP_0F, PP_66, shift_count_opcode(op, lane)), (dst, sx), X1);
            return Ok(());
        }
        let Some(shift) = lane_shift(op, lane) else {
            return Err(format!("{op:?} of 256-bit vector {lane:?} by unequal counts"));
        };
        let sx = self.fin(x, X0);
        let sy = self.wide_rm(y, X1);
        self.asm.vex(Vl::Y, shift, (dst, sx), sy);
        Ok(())
    }

    /// Emits the lanewise `op` of one 32-byte vector.
    pub(super) fn ymm_unary(&mut self, op: Unary, x: Value, result: Value) -> Result<(), String> {
        let Some((lane, _)) = self.ty(x).vec_shape() else {
            return Err(format!("non-vector type {:?} in ymm_unary", self.ty(x)));
        };
        let dst = self.fdst(result);
        match (op, vec_alu_opcode(Binary::Isub, lane)) {
            (Unary::Bnot, _) => {
                let sx = self.fin(x, X0);
                self.asm.vex(Vl::Y, PCMPEQD, (X1, X1), X1);
                self.asm.vex(Vl::Y, PXOR, (dst, sx), X1);
            }
            (Unary::Ineg, Some(sub)) if lane.is_int() => {
                self.zero_xmm(X1);
                let sx = self.wide_rm(x, X0);
                self.asm.vex(Vl::Y, VexOp::new(MAP_0F, PP_66, sub), (dst, X1), sx);
            }
            _ => return Err(format!("{op:?} of 256-bit vector {lane:?}")),
        }
        self.clobber_xmms(&[X0, X1]);
        self.fput(result, dst);
        Ok(())
    }

    /// Emits `op` when it reads or makes a 32-byte vector; `None` leaves it to the 16-byte emitter.
    pub(super) fn wide_op(&mut self, op: &Op, result: Value) -> Option<Result<(), String>> {
        match op {
            Op::TargetOp(id, args) => self.wide_target(*id, args.as_slice(), result),
            Op::Shuffle(a, b, ShuffleMask::Const(mask)) if self.ty(*a).is_wide_vec() => Some(self.wide_shuffle((*a, *b), mask, result)),
            Op::Insert(v, _, 0) if self.ty(*v).is_wide_vec() && self.is_zero_vec(*v) => None,
            Op::Insert(v, x, at) if self.ty(*v).is_wide_vec() => Some(self.wide_insert(*v, *x, *at, result)),
            Op::Shuffle(v, ..) | Op::Extract(v, _) | Op::Reduce(_, v) if self.ty(*v).is_wide_vec() => Some(Err(format!("{op:?} of a 256-bit vector"))),
            _ => None,
        }
    }

    fn wide_target(&mut self, id: TargetOpId, args: &[Value], result: Value) -> Option<Result<(), String>> {
        if let (TargetOpId::X64Pmovmskb256, &[v]) = (id, args) {
            let dst = self.dst(result);
            let src = self.fin(v, X0);
            self.asm.vex(Vl::Y, PMOVMSKB, (Xmm(dst as u8), Xmm(0)), src);
            self.clobber_xmms(&[X0]);
            self.put(result, dst);
            return Some(Ok(()));
        }
        let dst = self.fdst(result);
        match (id, byte_op(id), args) {
            (TargetOpId::X64Low128, _, &[v]) => {
                let src = self.wide_rm(v, dst);
                if !matches!(src, Rm::Reg(reg) if reg == dst.0) {
                    self.asm.vex(Vl::X, MOVDQU_LOAD, (dst, Xmm(0)), src);
                }
            }
            (TargetOpId::X64High128, _, &[v]) => {
                let half = self.ty(result);
                match self.wide_rm(v, X0) {
                    Rm::Mem(mem) => self.xmm_load(half, dst, mem.offset(half.bytes() as i32)),
                    Rm::Reg(src) => self.asm.vex_imm(Vl::Y, EXTRACTI128, (Xmm(src), Xmm(0)), dst, HIGH_HALF),
                }
            }
            (TargetOpId::X64Concat128, _, &[low, high]) => {
                let slow = self.fin(low, X0);
                let shigh = self.foperand(high, X1);
                self.asm.vex_imm(Vl::Y, INSERTI128, (dst, slow), shigh, HIGH_HALF);
            }
            (_, Some(op), &[x, y]) => {
                let sx = self.fin(x, X0);
                let sy = self.wide_rm(y, X1);
                self.asm.vex(Vl::Y, op, (dst, sx), sy);
            }
            _ => return None,
        }
        self.clobber_xmms(&[X0, X1]);
        self.fput(result, dst);
        Some(Ok(()))
    }

    fn wide_insert(&mut self, v: Value, x: Value, at: u8, result: Value) -> Result<(), String> {
        let Some((lane, count)) = self.ty(v).vec_shape() else {
            return Err(format!("non-vector type {:?} in wide_insert", self.ty(v)));
        };
        let dst = self.fdst(result);
        let reg = if lane.is_float() {
            let src = self.fin(x, X1);
            if lane == Lane::F64 { self.asm.movq_out(Reg::Rax, src) } else { self.asm.movd_out(Reg::Rax, src) }
            self.clobber_reg(Reg::Rax);
            Reg::Rax
        } else {
            self.reg_of(x).unwrap_or_else(|| {
                self.get(Reg::Rax, x, Ext::Zero);
                Reg::Rax
            })
        };
        let half = count / 2;
        let sv = self.fin(v, X0);
        if at < half {
            self.vpinsr(lane.int_mask(), (X1, sv), reg, at);
            self.asm.vex_imm(Vl::Y, PBLENDD, (dst, sv), X1, LOW_DWORDS);
        } else {
            self.asm.vex_imm(Vl::Y, EXTRACTI128, (sv, Xmm(0)), X1, HIGH_HALF);
            self.vpinsr(lane.int_mask(), (X1, X1), reg, at - half);
            self.asm.vex_imm(Vl::Y, INSERTI128, (dst, sv), X1, HIGH_HALF);
        }
        self.clobber_xmms(&[X0, X1]);
        self.fput(result, dst);
        Ok(())
    }

    fn wide_shuffle(&mut self, (a, b): (Value, Value), mask: &[u8; 16], result: Value) -> Result<(), String> {
        let Some((lane, count)) = self.ty(a).vec_shape() else {
            return Err(format!("non-vector type {:?} in wide_shuffle", self.ty(a)));
        };
        let dst = self.fdst(result);
        let lanes: Vec<u8> = mask[..usize::from(count)].iter().map(|&at| if a == b && at >= count { at - count } else { at }).collect();
        if lanes.iter().all(|&at| at == lanes[0]) && lanes[0] % count == 0 {
            let src = self.fin(if lanes[0] == 0 { a } else { b }, X0);
            self.asm.vex(Vl::Y, broadcast(lane), (dst, Xmm(0)), src);
        } else if let Ok(quads) = <[u8; QUADS]>::try_from(lanes.as_slice()) {
            self.permute_quads((a, b), quads, dst);
        } else {
            return Err(format!("shuffle {lanes:?} of 256-bit vector {lane:?}"));
        }
        self.clobber_xmms(&[X0, X1, X2]);
        self.fput(result, dst);
        Ok(())
    }

    fn permute_quads(&mut self, (a, b): (Value, Value), quads: [u8; QUADS], dst: Xmm) {
        let split = QUADS as u8;
        let one = if quads.iter().all(|&quad| quad < split) {
            Some((a, quads))
        } else if quads.iter().all(|&quad| quad >= split) {
            Some((b, quads.map(|quad| quad - split)))
        } else {
            None
        };
        if let Some((src, own)) = one {
            if own == [0, 1, 2, 3] {
                let from = self.fin(src, dst);
                if from != dst {
                    self.asm.vex(Vl::Y, MOVAPS, (dst, Xmm(0)), from);
                }
                return;
            }
            let from = self.wide_rm(src, X1);
            return self.asm.vex_imm(Vl::Y, PERMQ, (dst, Xmm(0)), from, quad_imm(own));
        }
        if quads.iter().enumerate().all(|(at, &quad)| usize::from(quad) % QUADS == at) {
            let first = self.fin(a, X0);
            let second = self.wide_rm(b, X1);
            return self.asm.vex_imm(Vl::Y, PBLENDD, (dst, first), second, blend_imm(quads));
        }
        if let Some(row) = PAIRED.iter().find(|row| row.quads == quads) {
            let (first, second) = if row.swap { (b, a) } else { (a, b) };
            let first = self.fin(first, X0);
            let second = self.wide_rm(second, X1);
            return match row.imm {
                Some(imm) => self.asm.vex_imm(Vl::Y, row.op, (dst, first), second, imm),
                None => self.asm.vex(Vl::Y, row.op, (dst, first), second),
            };
        }
        if quads[0] % 2 == 0 && quads[1] == quads[0] + 1 && quads[2] % 2 == 0 && quads[3] == quads[2] + 1 {
            let first = self.fin(a, X0);
            let second = self.wide_rm(b, X1);
            return self.asm.vex_imm(Vl::Y, PERM2I128, (dst, first), second, (quads[0] / 2) | ((quads[2] / 2) << 4));
        }
        let first = self.wide_rm(a, X1);
        self.asm.vex_imm(Vl::Y, PERMQ, (X1, Xmm(0)), first, quad_imm(quads));
        let second = self.wide_rm(b, X2);
        self.asm.vex_imm(Vl::Y, PERMQ, (X2, Xmm(0)), second, quad_imm(quads));
        self.asm.vex_imm(Vl::Y, PBLENDD, (dst, X1), X2, blend_imm(quads));
    }
}
