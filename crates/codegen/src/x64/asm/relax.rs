use super::{Asm, Label};
use crate::{Site, StencilReloc};

const NOPS: [&[u8]; 9] = [
    &[0x90],
    &[0x66, 0x90],
    &[0x0F, 0x1F, 0x00],
    &[0x0F, 0x1F, 0x40, 0x00],
    &[0x0F, 0x1F, 0x44, 0x00, 0x00],
    &[0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00],
    &[0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00],
    &[0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
    &[0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
];

fn emit_nops(out: &mut Vec<u8>, mut n: u32) {
    while n > 0 {
        let step = n.min(9) as usize;
        out.extend_from_slice(NOPS[step - 1]);
        n -= step as u32;
    }
}

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

#[derive(Clone, Copy)]
struct Event {
    at: u32,
    delta: i32,
    is_align: bool,
}

pub(crate) struct Moves(Vec<Event>);

impl Moves {
    fn of(fixups: &[Fixup], short: &[bool], aligns: &[(u32, u8)]) -> (Self, Vec<(u32, u32)>) {
        let mut events = Vec::with_capacity(fixups.len() + aligns.len());
        let mut pads = Vec::with_capacity(aligns.len());
        let mut delta = 0_i32;
        let (mut fi, mut ai) = (0, 0);
        loop {
            while fi < fixups.len() && !short[fi] {
                fi += 1;
            }
            let next_f = fixups.get(fi).map(|f| f.at + 4);
            let next_a = aligns.get(ai).map(|&(at, _)| at);
            match (next_f, next_a) {
                (Some(f_at), Some(a_at)) if a_at < f_at => {
                    let max_pad = u32::from(aligns[ai].1);
                    let cur = (i64::from(a_at) + i64::from(delta)) as u32;
                    let need = (16 - (cur & 15)) & 15;
                    if need > 0 && need <= max_pad {
                        delta += need as i32;
                        pads.push((a_at, need));
                        events.push(Event {
                            at: a_at,
                            delta,
                            is_align: true,
                        });
                    }
                    ai += 1;
                }
                (Some(f_at), _) => {
                    delta -= (fixups[fi].disp.opcode_len() + 2) as i32;
                    events.push(Event {
                        at: f_at,
                        delta,
                        is_align: false,
                    });
                    fi += 1;
                }
                (None, Some(a_at)) => {
                    let max_pad = u32::from(aligns[ai].1);
                    let cur = (i64::from(a_at) + i64::from(delta)) as u32;
                    let need = (16 - (cur & 15)) & 15;
                    if need > 0 && need <= max_pad {
                        delta += need as i32;
                        pads.push((a_at, need));
                        events.push(Event {
                            at: a_at,
                            delta,
                            is_align: true,
                        });
                    }
                    ai += 1;
                }
                (None, None) => break,
            }
        }
        (Self(events), pads)
    }

    pub fn map(&self, at: u32) -> u32 {
        let index = self.0.partition_point(|e| e.at <= at);
        let delta = index.checked_sub(1).map_or(0, |last| self.0[last].delta);
        (i64::from(at) + i64::from(delta)) as u32
    }

    fn map_end(&self, at: u32) -> u32 {
        let index = self
            .0
            .partition_point(|e| e.at < at || (e.at == at && !e.is_align));
        let delta = index.checked_sub(1).map_or(0, |last| self.0[last].delta);
        (i64::from(at) + i64::from(delta)) as u32
    }

    /// Moves block starts, exception sites and relocations to their offsets in the relaxed code.
    pub fn apply(&self, starts: &mut [u32], sites: &mut [Site], relocs: &mut [StencilReloc]) {
        starts.iter_mut().for_each(|at| *at = self.map(*at));
        for site in sites {
            site.ret = self.map_end(site.ret);
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
        i8::try_from(i64::from(moves.map(target)) - i64::from(moves.map_end(fixup.at + 4))).is_ok()
    }

    pub fn relax(&mut self) -> Moves {
        let mut short: Vec<bool> = self
            .fixups
            .iter()
            .map(|fixup| matches!(fixup.disp, Disp::Jmp | Disp::Jcc))
            .collect();
        let (moves, pads) = loop {
            let (moves, pads) = Moves::of(&self.fixups, &short, &self.aligns);
            let mut grew = false;
            for (fixup, short) in self.fixups.iter().zip(&mut short) {
                if *short && !self.reaches(fixup, &moves) {
                    *short = false;
                    grew = true;
                }
            }
            if !grew {
                break (moves, pads);
            }
        };
        let mut code = Vec::with_capacity(self.code.len() + pads.len() * 8);
        let (mut from, mut fi, mut pi) = (0usize, 0, 0);
        loop {
            while fi < self.fixups.len() && !short[fi] {
                fi += 1;
            }
            let next_f = self.fixups.get(fi).map(|f| f.at + 4);
            let next_p = pads.get(pi).map(|&(at, _)| at);
            match (next_f, next_p) {
                (Some(f_end), Some(p_at)) if p_at < f_end => {
                    let end = p_at as usize;
                    code.extend_from_slice(&self.code[from..end]);
                    emit_nops(&mut code, pads[pi].1);
                    from = end;
                    pi += 1;
                }
                (Some(f_end), _) => {
                    let fixup = self.fixups[fi];
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
                    from = f_end as usize;
                    fi += 1;
                }
                (None, Some(p_at)) => {
                    let end = p_at as usize;
                    code.extend_from_slice(&self.code[from..end]);
                    emit_nops(&mut code, pads[pi].1);
                    from = end;
                    pi += 1;
                }
                (None, None) => break,
            }
        }
        code.extend_from_slice(&self.code[from..]);
        self.code = code;
        for (fixup, &short) in self.fixups.iter_mut().zip(&short) {
            (fixup.at, fixup.disp) = if short {
                (moves.map_end(fixup.at + 4) - 1, Disp::Short)
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
        self.aligns.clear();
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
