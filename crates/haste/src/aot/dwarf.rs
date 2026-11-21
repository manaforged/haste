use std::path::PathBuf;

use rustc_hash::FxHashMap;

use crate::outline::source::file_key;

const LINE_BASE: i8 = -5;
const LINE_RANGE: u8 = 14;
const OPCODE_BASE: u8 = 13;
const STANDARD_LENGTHS: [u8; 12] = [0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1];

/// The DWARF 4 `.debug_line` section for `.text` at `text_va` of `text_len` bytes, from `.text` offsets paired with `Source::place` keys, naming files among `paths`.
pub(super) fn debug_line(text_va: u64, text_len: u32, lines: &[(u32, u64)], paths: &[PathBuf]) -> Vec<u8> {
    let by_key: FxHashMap<u32, &PathBuf> = paths.iter().map(|path| (file_key(path), path)).collect();
    let mut files: Vec<(&PathBuf, Vec<u32>)> = Vec::new();
    let mut index: FxHashMap<u32, u32> = FxHashMap::default();
    let mut rows = Vec::with_capacity(lines.len());
    for &(at, place) in lines {
        let Some(&path) = by_key.get(&((place >> 32) as u32)) else {
            rows.push((at, 0, 0));
            continue;
        };
        let file = *index.entry((place >> 32) as u32).or_insert_with(|| {
            files.push((path, line_starts(path)));
            files.len() as u32
        });
        let line = files[file as usize - 1].1.partition_point(|&start| start <= place as u32);
        rows.push((at, file, line as i64));
    }
    if files.is_empty() {
        return Vec::new();
    }
    rows.sort_by_key(|&(at, ..)| at);
    let mut program = Vec::new();
    let (mut file, mut line) = (1, 1);
    for (at, row_file, row_line) in rows {
        set_address(&mut program, text_va + u64::from(at));
        if row_file != 0 && row_file != file {
            program.push(4);
            uleb(&mut program, u64::from(row_file));
            file = row_file;
        }
        if row_line != line {
            program.push(3);
            sleb(&mut program, row_line - line);
            line = row_line;
        }
        program.push(1);
    }
    set_address(&mut program, text_va + u64::from(text_len));
    program.extend([0, 1, 1]);
    let mut head = vec![1, 1, 1, LINE_BASE as u8, LINE_RANGE, OPCODE_BASE];
    head.extend(STANDARD_LENGTHS);
    head.push(0);
    for (path, _) in &files {
        head.extend(path.as_os_str().as_encoded_bytes());
        head.extend([0, 0, 0, 0]);
    }
    head.push(0);
    let mut out = Vec::with_capacity(10 + head.len() + program.len());
    out.extend(((2 + 4 + head.len() + program.len()) as u32).to_le_bytes());
    out.extend(4u16.to_le_bytes());
    out.extend((head.len() as u32).to_le_bytes());
    out.extend(head);
    out.extend(program);
    out
}

fn line_starts(path: &PathBuf) -> Vec<u32> {
    let text = std::fs::read(path).unwrap_or_default();
    std::iter::once(0).chain(text.iter().enumerate().filter(|&(_, &byte)| byte == b'\n').map(|(at, _)| at as u32 + 1)).collect()
}

fn set_address(out: &mut Vec<u8>, address: u64) {
    out.extend([0, 9, 2]);
    out.extend(address.to_le_bytes());
}

fn uleb(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

fn sleb(out: &mut Vec<u8>, mut value: i64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if (value == 0 && byte & 0x40 == 0) || (value == -1 && byte & 0x40 != 0) {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}
