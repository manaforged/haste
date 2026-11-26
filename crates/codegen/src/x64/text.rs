use super::asm::{Alu, Asm, Mem, Reg, Size, Xmm};
use crate::asm_text::{Encoder, Symbols, integer};
use crate::reloc::StencilReloc;

const REGS: [(&str, Reg); 16] = [
    ("rax", Reg::Rax),
    ("rcx", Reg::Rcx),
    ("rdx", Reg::Rdx),
    ("rbx", Reg::Rbx),
    ("rsp", Reg::Rsp),
    ("rbp", Reg::Rbp),
    ("rsi", Reg::Rsi),
    ("rdi", Reg::Rdi),
    ("r8", Reg::R8),
    ("r9", Reg::R9),
    ("r10", Reg::R10),
    ("r11", Reg::R11),
    ("r12", Reg::R12),
    ("r13", Reg::R13),
    ("r14", Reg::R14),
    ("r15", Reg::R15),
];

/// The x86-64 encoder of Intel-syntax `global_asm!` text.
pub(crate) struct Text {
    asm: Asm,
    relocs: Vec<StencilReloc>,
    direct: bool,
}

#[derive(Clone, Copy)]
enum Operand {
    Reg(Reg),
    Xmm(Xmm),
    Mem(Mem),
    Imm(i32),
    Symbol(u32),
}

impl Text {
    /// An empty encoder whose `call {sym}` is PC-relative when `direct`, else through the call table slot.
    pub(crate) fn new(direct: bool) -> Self {
        Self {
            asm: Asm::default(),
            relocs: Vec::new(),
            direct,
        }
    }

    fn data(&mut self, mnemonic: &str, operands: &[Operand]) -> bool {
        match (mnemonic, operands) {
            ("mov", [Operand::Reg(dst), Operand::Reg(src)]) => self.asm.mov(*dst, *src),
            ("mov", [Operand::Mem(dst), Operand::Reg(src)]) => self.asm.store(Size::Q, *dst, *src),
            ("mov", [Operand::Reg(dst), Operand::Mem(src)]) => self.asm.load(Size::Q, *dst, *src),
            ("movdqu", [Operand::Mem(dst), Operand::Xmm(src)]) => {
                self.asm.sse(&[0xF3], &[0x0F, 0x7F], src.0, *dst)
            }
            ("movdqu", [Operand::Xmm(dst), Operand::Mem(src)]) => {
                self.asm.sse(&[0xF3], &[0x0F, 0x6F], dst.0, *src)
            }
            ("add", [Operand::Reg(dst), Operand::Imm(imm)]) => {
                self.asm.alu_imm(Alu::Add, *dst, *imm)
            }
            ("sub", [Operand::Reg(dst), Operand::Imm(imm)]) => {
                self.asm.alu_imm(Alu::Sub, *dst, *imm)
            }
            _ => return false,
        }
        true
    }

    fn control(&mut self, mnemonic: &str, operands: &[Operand]) -> bool {
        match (mnemonic, operands) {
            ("push", [Operand::Reg(reg)]) => self.asm.push(*reg),
            ("pop", [Operand::Reg(reg)]) => self.asm.pop(*reg),
            ("call", [Operand::Symbol(sym)]) => {
                let reloc = self.asm.call_symbol(*sym, self.direct);
                self.relocs.push(reloc);
            }
            ("call", [Operand::Reg(reg)]) => self.asm.call(*reg),
            ("jmp", [Operand::Reg(reg)]) => self.asm.jmp_reg(*reg),
            ("ret", []) => self.asm.bytes(&[0xC3]),
            _ => return false,
        }
        true
    }
}

fn register(text: &str) -> Option<Reg> {
    REGS.iter()
        .find(|(name, _)| *name == text)
        .map(|&(_, reg)| reg)
}

fn memory(inner: &str) -> Result<Mem, String> {
    let inner = inner.trim();
    let (base, disp) = match inner.find(|ch| ch == '+' || ch == '-') {
        Some(at) => (inner[..at].trim(), integer(&inner[at..].replace(' ', ""))?),
        None => (inner, 0),
    };
    let base = register(base).ok_or_else(|| format!("`[{inner}]` is not `[register + offset]`"))?;
    let disp = i32::try_from(disp)
        .map_err(|_| format!("the offset of `[{inner}]` does not fit 32 bits"))?;
    Ok(Mem::new(base, disp))
}

fn operand(text: &str, symbol: &Symbols<'_>) -> Result<Operand, String> {
    if text.starts_with('{') {
        return symbol(text).map(Operand::Symbol);
    }
    if let Some(inner) = text
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
    {
        return memory(inner).map(Operand::Mem);
    }
    if let Some(reg) = register(text) {
        return Ok(Operand::Reg(reg));
    }
    if let Some(index) = text
        .strip_prefix("xmm")
        .and_then(|index| index.parse::<u8>().ok())
        .filter(|&index| index < 16)
    {
        return Ok(Operand::Xmm(Xmm(index)));
    }
    i32::try_from(integer(text)?)
        .map(Operand::Imm)
        .map_err(|_| format!("`{text}` does not fit 32 bits"))
}

impl Encoder for Text {
    fn pos(&self) -> u32 {
        self.asm.pos()
    }

    fn pad(&mut self) {
        self.asm.bytes(&[0x90]);
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
        if self.data(mnemonic, &operands) || self.control(mnemonic, &operands) {
            return Ok(());
        }
        Err(format!("unsupported x86-64 instruction form `{mnemonic}`"))
    }

    fn finish(self) -> Result<(Vec<u8>, Vec<StencilReloc>), String> {
        Ok((self.asm.finish()?, self.relocs))
    }
}
