use std::hash::Hasher as _;

use rustc_hash::FxHasher;

const PAGE_SHIFT: u8 = 12;
const PAGE_SIZE: usize = 1 << PAGE_SHIFT;
const HASH_LEN: usize = 32;
const SUPER_HEADER: usize = 20;
const CD_HEADER: usize = 88;
const IDENT: &[u8] = b"a.out\0";

const CSMAGIC_CODEDIRECTORY: u32 = 0xFADE_0C02;
const CSMAGIC_EMBEDDED_SIGNATURE: u32 = 0xFADE_0CC0;
const CS_VERSION_EXECSEG: u32 = 0x0002_0400;
const CS_ADHOC_LINKER_SIGNED: u32 = 0x0002_0002;
const CS_HASHTYPE_SHA256: u8 = 2;
const CS_EXECSEG_MAIN_BINARY: u64 = 1;

const SHA256_IV: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

const SHA256_K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

fn compress(state: &mut [u32; 8], block: &[u8; 64]) {
    let mut w = [0u32; 64];
    for (i, chunk) in block.chunks_exact(4).enumerate() {
        w[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *state;
    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let t1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(SHA256_K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    for (dst, add) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *dst = dst.wrapping_add(add);
    }
}

fn sha256(data: &[u8]) -> [u8; HASH_LEN] {
    let mut state = SHA256_IV;
    let mut chunks = data.chunks_exact(64);
    for chunk in &mut chunks {
        let block: [u8; 64] = [
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            chunk[8], chunk[9], chunk[10], chunk[11], chunk[12], chunk[13], chunk[14], chunk[15],
            chunk[16], chunk[17], chunk[18], chunk[19], chunk[20], chunk[21], chunk[22], chunk[23],
            chunk[24], chunk[25], chunk[26], chunk[27], chunk[28], chunk[29], chunk[30], chunk[31],
            chunk[32], chunk[33], chunk[34], chunk[35], chunk[36], chunk[37], chunk[38], chunk[39],
            chunk[40], chunk[41], chunk[42], chunk[43], chunk[44], chunk[45], chunk[46], chunk[47],
            chunk[48], chunk[49], chunk[50], chunk[51], chunk[52], chunk[53], chunk[54], chunk[55],
            chunk[56], chunk[57], chunk[58], chunk[59], chunk[60], chunk[61], chunk[62], chunk[63],
        ];
        compress(&mut state, &block);
    }
    let rem = chunks.remainder();
    let mut tail = [0u8; 128];
    tail[..rem.len()].copy_from_slice(rem);
    tail[rem.len()] = 0x80;
    let block_count = if rem.len() + 9 <= 64 { 1 } else { 2 };
    let len_bits = (data.len() as u64) * 8;
    let len_at = block_count * 64 - 8;
    tail[len_at..len_at + 8].copy_from_slice(&len_bits.to_be_bytes());
    for i in 0..block_count {
        let off = i * 64;
        let block: [u8; 64] = [
            tail[off],
            tail[off + 1],
            tail[off + 2],
            tail[off + 3],
            tail[off + 4],
            tail[off + 5],
            tail[off + 6],
            tail[off + 7],
            tail[off + 8],
            tail[off + 9],
            tail[off + 10],
            tail[off + 11],
            tail[off + 12],
            tail[off + 13],
            tail[off + 14],
            tail[off + 15],
            tail[off + 16],
            tail[off + 17],
            tail[off + 18],
            tail[off + 19],
            tail[off + 20],
            tail[off + 21],
            tail[off + 22],
            tail[off + 23],
            tail[off + 24],
            tail[off + 25],
            tail[off + 26],
            tail[off + 27],
            tail[off + 28],
            tail[off + 29],
            tail[off + 30],
            tail[off + 31],
            tail[off + 32],
            tail[off + 33],
            tail[off + 34],
            tail[off + 35],
            tail[off + 36],
            tail[off + 37],
            tail[off + 38],
            tail[off + 39],
            tail[off + 40],
            tail[off + 41],
            tail[off + 42],
            tail[off + 43],
            tail[off + 44],
            tail[off + 45],
            tail[off + 46],
            tail[off + 47],
            tail[off + 48],
            tail[off + 49],
            tail[off + 50],
            tail[off + 51],
            tail[off + 52],
            tail[off + 53],
            tail[off + 54],
            tail[off + 55],
            tail[off + 56],
            tail[off + 57],
            tail[off + 58],
            tail[off + 59],
            tail[off + 60],
            tail[off + 61],
            tail[off + 62],
            tail[off + 63],
        ];
        compress(&mut state, &block);
    }
    let mut out = [0u8; HASH_LEN];
    for (i, word) in state.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

fn hash_slice_pair(bytes: &[u8]) -> (u64, u64) {
    let mut h1 = FxHasher::default();
    let mut h2 = FxHasher::default();
    h1.write(bytes);
    let mut chunks = bytes.chunks_exact(8);
    for c in &mut chunks {
        let word = u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
        h2.write_u64(word.swap_bytes());
    }
    let rem = chunks.remainder();
    if !rem.is_empty() {
        let mut tail = [0u8; 8];
        tail[..rem.len()].copy_from_slice(rem);
        h2.write_u64(u64::from_le_bytes(tail).swap_bytes());
    }
    (h1.finish(), h2.finish())
}

pub(super) fn deterministic_uuid(bytes: &[u8]) -> [u8; 16] {
    let workers = crate::jobs().min(bytes.len() / (512 * 1024));
    let (v1, v2) = if workers <= 1 {
        let (a, b) = hash_slice_pair(bytes);
        let mut h1 = FxHasher::default();
        let mut h2 = FxHasher::default();
        h1.write_u64(0x9E37_79B9_7F4A_7C15);
        h2.write_u64(0xBF58_476D_1CE4_E5B9);
        h1.write_usize(bytes.len());
        h2.write_usize(bytes.len() ^ 0x5555_5555_5555_5555);
        h1.write_u64(a);
        h2.write_u64(b);
        (h1.finish(), h2.finish())
    } else {
        let chunk_len = bytes.len().div_ceil(workers).next_multiple_of(8).max(8);
        let parts: Vec<(u64, u64)> = crate::sync::scope(|scope| {
            let handles: Vec<_> = bytes
                .chunks(chunk_len)
                .map(|c| scope.spawn(move || hash_slice_pair(c)))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or((0, 0)))
                .collect()
        });
        let mut h1 = FxHasher::default();
        let mut h2 = FxHasher::default();
        h1.write_u64(0x9E37_79B9_7F4A_7C15);
        h2.write_u64(0xBF58_476D_1CE4_E5B9);
        h1.write_usize(bytes.len());
        h2.write_usize(bytes.len() ^ 0x5555_5555_5555_5555);
        for (a, b) in parts {
            h1.write_u64(a);
            h2.write_u64(b);
        }
        (h1.finish(), h2.finish())
    };
    let mut uuid = [0u8; 16];
    uuid[..8].copy_from_slice(&v1.to_le_bytes());
    uuid[8..].copy_from_slice(&v2.to_le_bytes());
    uuid[6] = (uuid[6] & 0x0F) | 0x40;
    uuid[8] = (uuid[8] & 0x3F) | 0x80;
    uuid
}

pub(super) fn signature_padded_len(code_limit: usize) -> usize {
    let n_code_slots = code_limit.div_ceil(PAGE_SIZE);
    let cd_len = CD_HEADER + IDENT.len() + n_code_slots * HASH_LEN;
    (SUPER_HEADER + cd_len).next_multiple_of(16)
}

fn push_be_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_be_bytes());
}

fn push_be_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_be_bytes());
}

pub(super) fn build_ad_hoc_signature(image: &[u8], text_seg_size: usize) -> Vec<u8> {
    let code_limit = image.len();
    let n_code_slots = code_limit.div_ceil(PAGE_SIZE);
    let hash_off = CD_HEADER + IDENT.len();
    let cd_len = hash_off + n_code_slots * HASH_LEN;
    let total_len = SUPER_HEADER + cd_len;
    let padded_len = total_len.next_multiple_of(16);

    let mut out = Vec::with_capacity(padded_len);
    push_be_u32(&mut out, CSMAGIC_EMBEDDED_SIGNATURE);
    push_be_u32(&mut out, total_len as u32);
    push_be_u32(&mut out, 1);
    push_be_u32(&mut out, 0);
    push_be_u32(&mut out, SUPER_HEADER as u32);

    let (limit32, limit64) = match u32::try_from(code_limit) {
        Ok(v) => (v, 0u64),
        Err(_) => (0u32, code_limit as u64),
    };
    push_be_u32(&mut out, CSMAGIC_CODEDIRECTORY);
    push_be_u32(&mut out, cd_len as u32);
    push_be_u32(&mut out, CS_VERSION_EXECSEG);
    push_be_u32(&mut out, CS_ADHOC_LINKER_SIGNED);
    push_be_u32(&mut out, hash_off as u32);
    push_be_u32(&mut out, CD_HEADER as u32);
    push_be_u32(&mut out, 0);
    push_be_u32(&mut out, n_code_slots as u32);
    push_be_u32(&mut out, limit32);
    out.extend_from_slice(&[HASH_LEN as u8, CS_HASHTYPE_SHA256, 0, PAGE_SHIFT]);
    push_be_u32(&mut out, 0);
    push_be_u32(&mut out, 0);
    push_be_u32(&mut out, 0);
    push_be_u32(&mut out, 0);
    push_be_u64(&mut out, limit64);
    push_be_u64(&mut out, 0);
    push_be_u64(&mut out, text_seg_size as u64);
    push_be_u64(&mut out, CS_EXECSEG_MAIN_BINARY);
    out.extend_from_slice(IDENT);

    let slots_start = out.len();
    out.resize(padded_len, 0);
    let hashes = &mut out[slots_start..slots_start + n_code_slots * HASH_LEN];
    let workers = crate::jobs().min(n_code_slots / 16);
    if workers <= 1 {
        for (page_bytes, slot) in image
            .chunks(PAGE_SIZE)
            .zip(hashes.chunks_exact_mut(HASH_LEN))
        {
            slot.copy_from_slice(&sha256(page_bytes));
        }
    } else {
        let pages_per_worker = n_code_slots.div_ceil(workers).max(1);
        let bytes_per_worker = pages_per_worker * PAGE_SIZE;
        crate::sync::scope(|scope| {
            for (img_chunk, hash_chunk) in image
                .chunks(bytes_per_worker)
                .zip(hashes.chunks_mut(pages_per_worker * HASH_LEN))
            {
                scope.spawn(move || {
                    for (page_bytes, slot) in img_chunk
                        .chunks(PAGE_SIZE)
                        .zip(hash_chunk.chunks_exact_mut(HASH_LEN))
                    {
                        slot.copy_from_slice(&sha256(page_bytes));
                    }
                });
            }
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aot::{ExeSymbol, Executable, Symbol};
    use crate::host::HostSymbol;
    use haste_codegen::{Arch, Format, RelocKind, TARGETS};

    fn hex_digest(digest: [u8; 32]) -> String {
        use std::fmt::Write as _;
        let mut s = String::with_capacity(64);
        for b in digest {
            let _ = write!(s, "{b:02x}");
        }
        s
    }

    #[test]
    fn sha256_matches_fips_vectors() {
        assert_eq!(
            hex_digest(sha256(b"")),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            hex_digest(sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            hex_digest(sha256(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn macho_embedded_ad_hoc_signature_matches_slots() {
        for target in TARGETS.iter().filter(|t| t.format == Format::MachO) {
            let mut exe = Executable::new(target);
            let (stub, call_off, kind) = match target.arch {
                Arch::X64 => (vec![0xE8, 0, 0, 0, 0, 0xC3], 1u32, RelocKind::PcRel32),
                Arch::Arm64 => (
                    vec![0x00, 0x00, 0x00, 0x94, 0xC0, 0x03, 0x5F, 0xD6],
                    0u32,
                    RelocKind::Arm64Call26,
                ),
            };
            let start = exe.function(&stub, 16, &[], &[]);
            let main = exe.function(&stub, 16, &[], &[]);
            exe.entry(start, main);
            exe.patch(
                main + call_off,
                kind,
                Symbol::Host(HostSymbol::Dynamic {
                    name: "puts".into(),
                    lib: None,
                }),
            );
            exe.symbols(vec![ExeSymbol {
                offset: main,
                name: "main".into(),
                exported: false,
            }]);
            let bytes = exe.encode().expect("encode Mach-O");
            let ncmds = u32::from_le_bytes(bytes[16..20].try_into().expect("ncmds"));
            let mut cur = 32usize;
            let mut code_sig = None;
            for _ in 0..ncmds {
                let cmd = u32::from_le_bytes(bytes[cur..cur + 4].try_into().expect("cmd"));
                let size = u32::from_le_bytes(bytes[cur + 4..cur + 8].try_into().expect("cmdsize"))
                    as usize;
                if cmd == 0x1D {
                    code_sig = Some((
                        u32::from_le_bytes(bytes[cur + 8..cur + 12].try_into().expect("off"))
                            as usize,
                        u32::from_le_bytes(bytes[cur + 12..cur + 16].try_into().expect("sz"))
                            as usize,
                    ));
                }
                cur += size;
            }
            let (sig_off, sig_sz) = code_sig.expect("LC_CODE_SIGNATURE");
            assert_eq!(sig_off + sig_sz, bytes.len());
            let sb_magic =
                u32::from_be_bytes(bytes[sig_off..sig_off + 4].try_into().expect("sb magic"));
            assert_eq!(sb_magic, CSMAGIC_EMBEDDED_SIGNATURE);
            let cd_off = sig_off + SUPER_HEADER;
            let cd_magic =
                u32::from_be_bytes(bytes[cd_off..cd_off + 4].try_into().expect("cd magic"));
            assert_eq!(cd_magic, CSMAGIC_CODEDIRECTORY);
            let hash_off = cd_off
                + u32::from_be_bytes(bytes[cd_off + 16..cd_off + 20].try_into().expect("hoff"))
                    as usize;
            let n_slots =
                u32::from_be_bytes(bytes[cd_off + 28..cd_off + 32].try_into().expect("slots"))
                    as usize;
            assert_eq!(n_slots, sig_off.div_ceil(PAGE_SIZE));
            for (slot_idx, page) in bytes[..sig_off].chunks(PAGE_SIZE).enumerate() {
                let at = hash_off + slot_idx * HASH_LEN;
                assert_eq!(&bytes[at..at + HASH_LEN], &sha256(page));
            }
            if cfg!(target_os = "macos") {
                let tmp = std::env::temp_dir().join(format!(
                    "haste-codesign-test-{}-{}",
                    std::process::id(),
                    target.triple
                ));
                exe.write(&tmp).expect("write Mach-O");
                let status = std::process::Command::new("codesign")
                    .args(["--verify", "--strict"])
                    .arg(&tmp)
                    .status()
                    .expect("run codesign --verify");
                drop(std::fs::remove_file(&tmp));
                assert!(
                    status.success(),
                    "codesign verify failed for {}",
                    target.triple
                );
            }
        }
    }
}
