use super::{Asm, Label};
use crate::{Site, StencilReloc};

/// How the displacement at a fixup is encoded.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Disp {
    Rip,
    Jmp,
    Jcc,
    Short,
}

impl Disp {
    fn opcode_len(self) -> u32 {
        match self {
            Self::Jcc => 2,
            Self::Jmp | Self::Short => 1,
            Self::Rip => 0,
        }
    }

    fn width(self) -> u32 {
        if self == Self::Short { 1 } else { 4 }
    }
}

/// A displacement at `at` that `finish` fills with the distance from its end to `label`.
#[derive(Clone, Copy)]
pub(crate) struct Fixup {
    pub(super) at: u32,
    pub(super) label: Label,
    pub(super) disp: Disp,
}

/// Where offsets taken before `relax` land after it: the end of each shortened branch and the bytes removed up to it.
pub(crate) struct Moves(Vec<(u32, u32)>);

impl Moves {
    fn of(fixups: &[Fixup], short: &[bool]) -> Self {
        let mut removed = 0;
        let ends = fixups
            .iter()
            .zip(short)
            .filter(|(_, short)| **short)
            .map(|(fixup, _)| {
                removed += fixup.disp.opcode_len() + 2;
                (fixup.at + 4, removed)
            });
        Self(ends.collect())
    }

    /// The offset in the relaxed code of `at`, an instruction boundary taken before relaxing.
    pub fn map(&self, at: u32) -> u32 {
        let index = self.0.partition_point(|&(end, _)| end <= at);
        at - index.checked_sub(1).map_or(0, |last| self.0[last].1)
    }

    /// Moves block starts, exception sites and relocations to their offsets in the relaxed code.
    pub fn apply(&self, starts: &mut [u32], sites: &mut [Site], relocs: &mut [StencilReloc]) {
        starts.iter_mut().for_each(|at| *at = self.map(*at));
        for site in sites {
            site.ret = self.map(site.ret);
            site.handler = self.map(site.handler);
        }
        relocs
            .iter_mut()
            .for_each(|reloc| reloc.offset = self.map(reloc.offset));
    }
}

impl Asm {
    fn place(&self, label: Label) -> Result<u32, String> {
        self.labels[label.0 as usize]
            .ok_or_else(|| "a jump to a label that was never placed".to_owned())
    }

    fn reaches(&self, fixup: &Fixup, moves: &Moves) -> bool {
        let Ok(target) = self.place(fixup.label) else {
            return false;
        };
        i8::try_from(i64::from(moves.map(target)) - i64::from(moves.map(fixup.at + 4))).is_ok()
    }

    /// Rewrites each `jmp` and `jcc` whose target lies within a signed byte in its two-byte form, and returns where earlier offsets moved.
    pub fn relax(&mut self) -> Moves {
        let mut short: Vec<bool> = self
            .fixups
            .iter()
            .map(|fixup| matches!(fixup.disp, Disp::Jmp | Disp::Jcc))
            .collect();
        let moves = loop {
            let moves = Moves::of(&self.fixups, &short);
            let mut grew = false;
            for (fixup, short) in self.fixups.iter().zip(&mut short) {
                if *short && !self.reaches(fixup, &moves) {
                    *short = false;
                    grew = true;
                }
            }
            if !grew {
                break moves;
            }
        };
        let mut code = Vec::with_capacity(self.code.len());
        let mut from = 0;
        for (fixup, _) in self.fixups.iter().zip(&short).filter(|(_, short)| **short) {
            let start = (fixup.at - fixup.disp.opcode_len()) as usize;
            code.extend_from_slice(&self.code[from..start]);
            code.extend([
                if fixup.disp == Disp::Jmp {
                    0xEB
                } else {
                    self.code[start + 1] - 0x10
                },
                0,
            ]);
            from = fixup.at as usize + 4;
        }
        code.extend_from_slice(&self.code[from..]);
        self.code = code;
        for (fixup, &short) in self.fixups.iter_mut().zip(&short) {
            (fixup.at, fixup.disp) = if short {
                (moves.map(fixup.at + 4) - 1, Disp::Short)
            } else {
                (moves.map(fixup.at), fixup.disp)
            };
        }
        self.labels
            .iter_mut()
            .flatten()
            .for_each(|at| *at = moves.map(*at));
        self.table_fixups
            .iter_mut()
            .for_each(|(at, _, _)| *at = moves.map(*at));
        self.tail = None;
        moves
    }

    pub fn finish(mut self) -> Result<Vec<u8>, String> {
        for Fixup { at, label, disp } in std::mem::take(&mut self.fixups) {
            let relative = i64::from(self.place(label)?) - i64::from(at + disp.width());
            let at = at as usize;
            if disp == Disp::Short {
                self.code[at] =
                    i8::try_from(relative).map_err(|_| "a short jump out of range")? as u8;
            } else {
                self.code[at..at + 4].copy_from_slice(&(relative as i32).to_le_bytes());
            }
        }
        for (at, base, target) in std::mem::take(&mut self.table_fixups) {
            let base = self.labels[base.0 as usize].ok_or("a jump table without a base label")?;
            let relative = i64::from(self.place(target)?) - i64::from(base);
            let at = at as usize;
            self.code[at..at + 4].copy_from_slice(&(relative as i32).to_le_bytes());
        }
        Ok(self.code)
    }
}
