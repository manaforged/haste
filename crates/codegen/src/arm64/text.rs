use super::asm::{Asm, FP, LR, Reg, SP, Size, Vreg};
use crate::asm_text::{Encoder, Symbols, integer};
use crate::reloc::StencilReloc;

const IP1: Reg = Reg(17);
const STP_Q: u32 = 0xAD00_0000;
const LDP_Q: u32 = 0xAD40_0000;

/// The A64 encoder of `global_asm!` text.
pub(crate) struct Text {
    asm: Asm,
    relocs: Vec<StencilReloc>,
    direct: bool,
}

#[derive(Clone, Copy)]
enum Address {
    Offset(Reg, i64),
    Pre(Reg, i64),
}

#[derive(Clone, Copy)]
enum Operand {
    X(Reg),
    Sp,
    Q(Vreg),
    Imm(i64),
    Symbol(u32),
    Addr(Address),
}

impl Text {
    /// An empty encoder whose `bl {sym}` is a direct branch when `direct`, else a call through the call table slot in x17.
    pub(crate) fn new(direct: bool) -> Self {
        Self {
            asm: Asm::default(),
            relocs: Vec::new(),
            direct,
        }
    }

    fn pair(&mut self, mnemonic: &str, operands: &[Operand]) -> Result<bool, String> {
        let load = match mnemonic {
            "stp" => false,
            "ldp" => true,
            _ => return Ok(false),
        };
        match (load, operands) {
            (
                false,
                [
                    Operand::X(a),
                    Operand::X(b),
                    Operand::Addr(Address::Pre(base, offset)),
                ],
            ) => self.asm.stp_pre(*a, *b, *base, scaled(*offset, 8)?),
            (
                true,
                [
                    Operand::X(a),
                    Operand::X(b),
                    Operand::Addr(Address::Offset(base, 0)),
                    Operand::Imm(offset),
                ],
            ) => self.asm.ldp_post(*a, *b, *base, scaled(*offset, 8)?),
            (
                false,
                [
                    Operand::X(a),
                    Operand::X(b),
                    Operand::Addr(Address::Offset(base, offset)),
                ],
            ) => self.asm.stp_offset(*a, *b, *base, scaled(*offset, 8)?),
            (
                true,
                [
                    Operand::X(a),
                    Operand::X(b),
                    Operand::Addr(Address::Offset(base, offset)),
                ],
            ) => self.asm.ldp_offset(*a, *b, *base, scaled(*offset, 8)?),
            (
                _,
                [
                    Operand::Q(a),
                    Operand::Q(b),
                    Operand::Addr(Address::Offset(base, offset)),
                ],
            ) => {
                self.asm.pair_offset(
                    if load { LDP_Q } else { STP_Q },
                    16,
                    a.0,
                    b.0,
                    *base,
                    scaled(*offset, 16)?,
                );
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn single(&mut self, mnemonic: &str, operands: &[Operand]) -> Result<bool, String> {
        match (mnemonic, operands) {
            ("mov", [Operand::X(dst), Operand::X(src)]) => self.asm.mov(*dst, *src),
            ("mov", [dst, src]) => self.asm.mov_sp(base(*dst)?, base(*src)?),
            ("add" | "sub", [dst, src, Operand::Imm(imm)]) => self.asm.add_imm(
                mnemonic == "sub",
                false,
                Size::X,
                base(*dst)?,
                base(*src)?,
                add_imm(*imm)?,
            ),
            ("bl", [Operand::Symbol(sym)]) => {
                let reloc = self.asm.call_symbol(*sym, self.direct, IP1);
                self.relocs.push(reloc);
            }
            ("blr", [Operand::X(reg)]) => self.asm.blr(*reg),
            ("br", [Operand::X(reg)]) => self.asm.br(*reg),
            ("ret", []) => self.asm.ret(),
            _ => return Ok(false),
        }
        Ok(true)
    }
}

fn base(operand: Operand) -> Result<Reg, String> {
    match operand {
        Operand::X(reg) => Ok(reg),
        Operand::Sp => Ok(SP),
        _ => Err("takes x0..x30 or sp".to_owned()),
    }
}

fn scaled(offset: i64, step: i64) -> Result<i32, String> {
    if offset % step != 0 || !(-64 * step..64 * step).contains(&offset) {
        return Err(format!(
            "pair offset {offset} is not a multiple of {step} within 7 signed bits"
        ));
    }
    Ok(offset as i32)
}

fn add_imm(imm: i64) -> Result<u32, String> {
    let fits = (0..4096).contains(&imm) || (imm % 4096 == 0 && (0..1 << 24).contains(&imm));
    if fits {
        Ok(imm as u32)
    } else {
        Err(format!("immediate {imm} does not fit add or sub"))
    }
}

fn register(text: &str) -> Option<Operand> {
    let index = |prefix: &str, limit: u8| {
        text.strip_prefix(prefix)
            .and_then(|number| number.parse::<u8>().ok())
            .filter(|&number| number < limit)
    };
    match text {
        "sp" => Some(Operand::Sp),
        "fp" => Some(Operand::X(FP)),
        "lr" => Some(Operand::X(LR)),
        _ => index("x", 31)
            .map(|number| Operand::X(Reg(number)))
            .or_else(|| index("q", 32).map(|number| Operand::Q(Vreg(number)))),
    }
}

fn address(text: &str) -> Result<Address, String> {
    let (body, pre) = text
        .strip_suffix('!')
        .map_or((text, false), |body| (body, true));
    let inner = body
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .ok_or_else(|| format!("`{text}` is not an address"))?;
    let (base_text, offset) = inner
        .split_once(',')
        .map_or((inner, Ok(0)), |(base_text, offset)| {
            (base_text, integer(offset.trim()))
        });
    let base = register(base_text.trim())
        .ok_or_else(|| format!("`{text}` has no base register"))
        .and_then(base)?;
    let offset = offset?;
    Ok(if pre {
        Address::Pre(base, offset)
    } else {
        Address::Offset(base, offset)
    })
}

fn operand(text: &str, symbol: &Symbols<'_>) -> Result<Operand, String> {
    if text.starts_with('{') {
        return symbol(text).map(Operand::Symbol);
    }
    if text.starts_with('[') {
        return address(text).map(Operand::Addr);
    }
    match register(text) {
        Some(reg) => Ok(reg),
        None => integer(text).map(Operand::Imm),
    }
}

impl Encoder for Text {
    fn pos(&self) -> u32 {
        self.asm.pos()
    }

    fn pad(&mut self) {
        self.asm.word(0xD503_201F);
    }

    fn instruction(
        &mut self,
        mnemonic: &str,
        operands: &[&str],
        symbol: &Symbols<'_>,
    ) -> Result<(), String> {
        let operands = operands
            .iter()
            .map(|text| operand(text, symbol))
            .collect::<Result<Vec<_>, _>>()?;
        if self.pair(mnemonic, &operands)? || self.single(mnemonic, &operands)? {
            return Ok(());
        }
        Err(format!("unsupported A64 instruction form `{mnemonic}`"))
    }

    fn finish(self) -> Result<(Vec<u8>, Vec<StencilReloc>), String> {
        Ok((self.asm.finish()?, self.relocs))
    }
}
