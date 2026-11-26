use super::asm::{Alu, Asm, Cond, Label, Mem, Reg, Size, Xmm};
use crate::asm_text::{Encoder, LocalLabels, Symbols, integer, local_target};
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
    locals: LocalLabels<Label>,
}

#[derive(Clone, Copy)]
enum Operand {
    Reg(Reg),
    Xmm(Xmm),
    Mem(Mem),
    Imm(i32),
    Symbol(u32),
    Local(u32, bool),
}

impl Text {
    /// An empty encoder whose `call {sym}` is PC-relative when `direct`, else through the call table slot.
    pub(crate) fn new(direct: bool) -> Self {
        Self {
            asm: Asm::default(),
            relocs: Vec::new(),
            direct,
            locals: LocalLabels::default(),
        }
    }

    fn local_label(&mut self, id: u32, forward: bool) -> Result<Label, String> {
        self.locals.target(id, forward, || self.asm.label())
    }

    fn data(&mut self, mnemonic: &str, operands: &[Operand]) -> bool {
        match (mnemonic, operands) {
            ("mov", [Operand::Reg(dst), Operand::Reg(src)]) => self.asm.mov(*dst, *src),
            ("mov", [Operand::Mem(dst), Operand::Reg(src)]) => self.asm.store(Size::Q, *dst, *src),
            ("mov", [Operand::Mem(dst), Operand::Imm(imm)]) => {
                self.asm.store_imm_sized(Size::Q, *dst, *imm)
            }
            ("mov", [Operand::Reg(dst), Operand::Mem(src)]) => self.asm.load(Size::Q, *dst, *src),
            ("test", [Operand::Reg(a), Operand::Reg(b)]) => self.asm.test(*a, *b),
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

    fn control(&mut self, mnemonic: &str, operands: &[Operand]) -> Result<bool, String> {
        match (mnemonic, operands) {
            ("push", [Operand::Reg(reg)]) => self.asm.push(*reg),
            ("pop", [Operand::Reg(reg)]) => self.asm.pop(*reg),
            ("call", [Operand::Symbol(sym)]) => {
                let reloc = self.asm.call_symbol(*sym, self.direct);
                self.relocs.push(reloc);
            }
            ("call", [Operand::Reg(reg)]) => self.asm.call(*reg),
            ("jmp", [Operand::Reg(reg)]) => self.asm.jmp_reg(*reg),
            ("jmp", [Operand::Local(id, forward)]) => {
                let label = self.local_label(*id, *forward)?;
                self.asm.jmp(label);
            }
            ("jz" | "je", [Operand::Local(id, forward)]) => {
                let label = self.local_label(*id, *forward)?;
                self.asm.jcc(Cond::E, label);
            }
            ("jnz" | "jne", [Operand::Local(id, forward)]) => {
                let label = self.local_label(*id, *forward)?;
                self.asm.jcc(Cond::Ne, label);
            }
            ("ret", []) => self.asm.bytes(&[0xC3]),
            _ => return Ok(false),
        }
        Ok(true)
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
    let mem_text = text.strip_prefix("qword ptr ").map_or(text, str::trim);
    if let Some(inner) = mem_text
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
    {
        return memory(inner).map(Operand::Mem);
    }
    if let Some((id, forward)) = local_target(text) {
        return Ok(Operand::Local(id, forward));
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

    fn define_local(&mut self, id: u32) {
        let label = self.locals.define(id, || self.asm.label());
        self.asm.bind(label);
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
        if self.data(mnemonic, &operands) || self.control(mnemonic, &operands)? {
            return Ok(());
        }
        Err(format!("unsupported x86-64 instruction form `{mnemonic}`"))
    }

    fn finish(self) -> Result<(Vec<u8>, Vec<StencilReloc>), String> {
        self.locals.finish()?;
        Ok((self.asm.finish()?, self.relocs))
    }
}
