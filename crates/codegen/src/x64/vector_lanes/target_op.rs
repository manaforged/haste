use crate::alloc::Location;
use crate::ir::{TargetOpId, Value};
use crate::x64::Emit;
use crate::x64::asm::Xmm;
use crate::x64::float::{X0, X1, X2};

#[derive(Clone, Copy)]
enum TargetOpForm {
    Unary,
    Binary,
    BinaryImm(u8),
    Sha256Rnds2,
}

struct TargetOpSpec {
    id: TargetOpId,
    form: TargetOpForm,
    prefix: &'static [u8],
    opcode: &'static [u8],
}

const X64_TARGET_OPS: &[TargetOpSpec] = &[
    TargetOpSpec {
        id: TargetOpId::X64Sha1NextE,
        form: TargetOpForm::Binary,
        prefix: &[],
        opcode: &[0x0F, 0x38, 0xC8],
    },
    TargetOpSpec {
        id: TargetOpId::X64Sha1Msg1,
        form: TargetOpForm::Binary,
        prefix: &[],
        opcode: &[0x0F, 0x38, 0xC9],
    },
    TargetOpSpec {
        id: TargetOpId::X64Sha1Msg2,
        form: TargetOpForm::Binary,
        prefix: &[],
        opcode: &[0x0F, 0x38, 0xCA],
    },
    TargetOpSpec {
        id: TargetOpId::X64Sha256Rnds2,
        form: TargetOpForm::Sha256Rnds2,
        prefix: &[],
        opcode: &[0x0F, 0x38, 0xCB],
    },
    TargetOpSpec {
        id: TargetOpId::X64Sha256Msg1,
        form: TargetOpForm::Binary,
        prefix: &[],
        opcode: &[0x0F, 0x38, 0xCC],
    },
    TargetOpSpec {
        id: TargetOpId::X64Sha256Msg2,
        form: TargetOpForm::Binary,
        prefix: &[],
        opcode: &[0x0F, 0x38, 0xCD],
    },
    TargetOpSpec {
        id: TargetOpId::X64AesEnc,
        form: TargetOpForm::Binary,
        prefix: &[0x66],
        opcode: &[0x0F, 0x38, 0xDC],
    },
    TargetOpSpec {
        id: TargetOpId::X64AesEncLast,
        form: TargetOpForm::Binary,
        prefix: &[0x66],
        opcode: &[0x0F, 0x38, 0xDD],
    },
    TargetOpSpec {
        id: TargetOpId::X64AesDec,
        form: TargetOpForm::Binary,
        prefix: &[0x66],
        opcode: &[0x0F, 0x38, 0xDE],
    },
    TargetOpSpec {
        id: TargetOpId::X64AesDecLast,
        form: TargetOpForm::Binary,
        prefix: &[0x66],
        opcode: &[0x0F, 0x38, 0xDF],
    },
    TargetOpSpec {
        id: TargetOpId::X64AesImc,
        form: TargetOpForm::Unary,
        prefix: &[0x66],
        opcode: &[0x0F, 0x38, 0xDB],
    },
    TargetOpSpec {
        id: TargetOpId::X64Psadbw,
        form: TargetOpForm::Binary,
        prefix: &[0x66],
        opcode: &[0x0F, 0xF6],
    },
    TargetOpSpec {
        id: TargetOpId::X64ClmulLowLow,
        form: TargetOpForm::BinaryImm(0x00),
        prefix: &[0x66],
        opcode: &[0x0F, 0x3A, 0x44],
    },
    TargetOpSpec {
        id: TargetOpId::X64ClmulHighLow,
        form: TargetOpForm::BinaryImm(0x01),
        prefix: &[0x66],
        opcode: &[0x0F, 0x3A, 0x44],
    },
    TargetOpSpec {
        id: TargetOpId::X64ClmulLowHigh,
        form: TargetOpForm::BinaryImm(0x10),
        prefix: &[0x66],
        opcode: &[0x0F, 0x3A, 0x44],
    },
    TargetOpSpec {
        id: TargetOpId::X64ClmulHighHigh,
        form: TargetOpForm::BinaryImm(0x11),
        prefix: &[0x66],
        opcode: &[0x0F, 0x3A, 0x44],
    },
];

impl Emit<'_> {
    fn cached_fspill(&self, value: Value) -> Option<Xmm> {
        if !matches!(self.location(value), Location::Slot) {
            return None;
        }
        let disp = self.frame.values[value.index()];
        let ty = self.ty(value);
        self.spills.cached_fspill(disp, ty)
    }

    fn emit_target_binary(
        &mut self,
        spec: &TargetOpSpec,
        (a, b, k): (Value, Value, Option<Value>),
        imm: Option<u8>,
        result: Value,
    ) {
        let dst = if k.is_some() {
            self.xmm_of(result).unwrap_or(X2)
        } else {
            self.fdst(result)
        };
        let mut got_k = k.is_none();
        let mut got_a = false;
        if let Some(kv) = k
            && self.cached_fspill(kv).is_some()
        {
            self.fget(X0, kv);
            got_k = true;
        } else if self.cached_fspill(a).is_some() && (self.xmm_of(b) != Some(dst) || a == b) {
            self.fget(dst, a);
            got_a = true;
        }
        let mut src_b = match self.cached_fspill(b) {
            Some(cached) if cached != dst && (k.is_none() || cached != X0) => cached,
            _ => self.fin(b, X1),
        };
        if src_b == dst && a != b {
            self.asm.movaps(X1, src_b);
            self.clobber_xmm(X1);
            src_b = X1;
        }
        if let Some(kv) = k
            && !got_k
        {
            self.fget(X0, kv);
        }
        if !got_a {
            self.fget(dst, a);
        }
        match imm {
            Some(imm) => self
                .asm
                .sse_imm(spec.prefix, spec.opcode, dst.0, src_b, imm),
            None => self.asm.sse(spec.prefix, spec.opcode, dst.0, src_b),
        }
        self.clobber_xmm(dst);
        self.fput(result, dst);
    }

    pub(super) fn vector_target_op(
        &mut self,
        id: TargetOpId,
        args: &[Value],
        result: Value,
    ) -> Result<(), String> {
        let spec = X64_TARGET_OPS
            .iter()
            .find(|s| s.id == id)
            .ok_or_else(|| format!("unsupported target op {id:?}"))?;
        match spec.form {
            TargetOpForm::Unary => {
                let [x] = *args else {
                    return Err(format!("{id:?} expects 1 arg"));
                };
                let dst = self.fdst(result);
                let src = self.fin(x, X0);
                self.asm.sse(spec.prefix, spec.opcode, dst.0, src);
                self.clobber_xmm(dst);
                self.fput(result, dst);
            }
            TargetOpForm::Binary | TargetOpForm::BinaryImm(_) => {
                let [x, y] = *args else {
                    return Err(format!("{id:?} expects 2 args"));
                };
                let imm = if let TargetOpForm::BinaryImm(imm) = spec.form {
                    Some(imm)
                } else {
                    None
                };
                self.emit_target_binary(spec, (x, y, None), imm, result);
            }
            TargetOpForm::Sha256Rnds2 => {
                let [a, b, k] = *args else {
                    return Err(format!("{id:?} expects 3 args"));
                };
                self.emit_target_binary(spec, (a, b, Some(k)), None, result);
            }
        }
        Ok(())
    }
}
