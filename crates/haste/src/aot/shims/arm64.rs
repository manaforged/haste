use anyhow::Result;
use haste_codegen::{Arch, Os};

use super::{Emit, Shim, arm64_fwd, arm64_here, arm64_jcc, arm64_patch, emit_arm64_adrp_add};
use crate::aot::{Buf, SectionAddrs, tls, unwind, wide};

fn emit_arm64_tls_get(c: &Emit<'_>, key_va: u64, out: &mut Buf) -> Result<()> {
    emit_arm64_adrp_add(c.at(out), key_va, 21, out);
    let ready_fast = arm64_jcc(&[0x88DF_FEA8, 0x7100_091F], 0x5400_0000, out);
    out.u32(0x5280_002A);
    let init_loop = out.pos();
    out.u32(0x885F_FEA8);
    let already_init = arm64_fwd(0x3500_0008, out);
    out.u32(0x8809_FEAA);
    let retry_cas = arm64_fwd(0x3500_0009, out);
    arm64_patch(retry_cas, init_loop, out);
    if c.win {
        c.call("TlsAlloc", out)?;
        out.u32(0xB900_0AA0);
    } else {
        c.words_call(&[0x9100_22A0, 0xAA1F_03E1], "pthread_key_create", out)?;
    }
    let ready_create = arm64_jcc(&[0x5280_0048, 0x889F_FEA8], 0x1400_0000, out);
    arm64_here(&[already_init], out);
    let ready_wait = arm64_jcc(&[0x7100_091F], 0x5400_0000, out);
    c.call(c.pick("SwitchToThread", "sched_yield"), out)?;
    let back = arm64_fwd(0x1400_0000, out);
    arm64_patch(back, init_loop, out);
    arm64_here(&[ready_fast, ready_create, ready_wait], out);
    c.words_call(
        &[if c.win { 0xB940_0AA0 } else { 0xF940_06A0 }],
        c.pick("TlsGetValue", "pthread_getspecific"),
        out,
    )
}

pub(in crate::aot) fn emit_arm64_unwind_state(
    c: &Emit<'_>,
    addrs: SectionAddrs,
    out: &mut Buf,
) -> Result<()> {
    if addrs.tlv_desc != 0 {
        tls::emit_arm64_tlv_unwind_state(c, addrs.tlv_desc, addrs.unwind_tls, out);
        return Ok(());
    }
    emit_arm64_tls_get(c, addrs.thread_key + 16, out)?;
    let hit = arm64_fwd(0xB500_0000, out);
    const _: () = assert!(tls::UNWIND_STATE_BYTES <= 0xFFFF);
    let mov_x1_size = 0xD280_0001_u32 | ((tls::UNWIND_STATE_BYTES as u32) << 5);
    c.words_call(&[0xD280_0020, mov_x1_size], "calloc", out)?;
    let fail = arm64_fwd(0xB400_0000, out);
    c.words_call(
        &[
            0xAA00_03F3,
            if c.win { 0xB940_0AA0 } else { 0xF940_06A0 },
            0xAA13_03E1,
        ],
        c.pick("TlsSetValue", "pthread_setspecific"),
        out,
    )?;
    out.u32(0xAA13_03E0);
    arm64_here(&[hit, fail], out);
    Ok(())
}

fn emit_arm64_unlink_x20(out: &mut Buf) {
    let no_prev = arm64_jcc(&[0xF940_0689, 0xF940_0A8A], 0xB400_0009, out);
    out.u32(0xF900_092A);
    arm64_here(&[no_prev], out);
    let is_tail = arm64_fwd(0xB400_000A, out);
    let after_next = arm64_jcc(&[0xF900_0549], 0x1400_0000, out);
    arm64_here(&[is_tail], out);
    out.u32(0xF900_0669);
    arm64_here(&[after_next], out);
    out.words(&[
        0xF900_069F,
        0xF900_0A9F,
        0x3940_0288,
        0x121E_7908,
        0x3900_0288,
    ]);
}

fn emit_arm64_drop_op(c: &Emit<'_>, push: bool, addrs: SectionAddrs, out: &mut Buf) -> Result<()> {
    let ret_null = arm64_fwd(0xB400_0000, out);
    let ret_unlinked = if push {
        None
    } else {
        Some(arm64_jcc(&[0x3940_0008, 0x721F_011F], 0x5400_0000, out))
    };
    out.words(&[0xA9BD_7BFD, 0xA901_53F3, 0xF900_13F5, 0xAA00_03F4]);
    emit_arm64_unwind_state(c, addrs, out)?;
    let no_state = arm64_fwd(0xB400_0000, out);
    out.u32(0xAA00_03F3);
    if push {
        let not_linked = arm64_jcc(&[0x3940_0288, 0x721F_011F], 0x5400_0000, out);
        emit_arm64_unlink_x20(out);
        arm64_here(&[not_linked], out);
        let no_head = arm64_jcc(
            &[
                0xF940_0668,
                0xF900_0688,
                0xF900_0A9F,
                0x3940_0289,
                0x321F_0129,
                0x3900_0289,
            ],
            0xB400_0008,
            out,
        );
        out.u32(0xF900_0914);
        arm64_here(&[no_head], out);
        out.u32(0xF900_0674);
    } else {
        emit_arm64_unlink_x20(out);
    }
    arm64_here(&[no_state], out);
    out.words(&[0xF940_13F5, 0xA941_53F3, 0xA8C3_7BFD]);
    arm64_here(&[ret_null], out);
    if let Some(lbl) = ret_unlinked {
        arm64_here(&[lbl], out);
    }
    out.u32(0xD65F_03C0);
    Ok(())
}

fn emit_arm64_thread_cell(c: &Emit<'_>, key_va: u64, out: &mut Buf) -> Result<()> {
    out.words(&[
        0xD101_43FF,
        0xA900_7BFD,
        0xA901_53F3,
        0xA902_5BF5,
        0xA903_63F7,
        0xAA00_03F3,
        0xAA02_03F8,
        0xF100_003F,
        0xD280_0029,
        0x9A89_1034,
    ]);
    emit_arm64_tls_get(c, key_va, out)?;
    out.u32(0xAA00_03F6);
    let grow_null = arm64_fwd(0xB400_0016, out);
    let grow_cap = arm64_jcc(&[0xF940_02C8, 0xEB08_027F], 0x5400_0002, out);
    let hit = arm64_jcc(&[0x8B13_0EC8, 0xF940_0500], 0xB500_0000, out);
    let cell_skip = arm64_fwd(0x1400_0000, out);
    arm64_here(&[grow_null, grow_cap], out);
    c.words_call(
        &[
            0xD280_0008,
            0xB400_0056,
            0xF940_02C8,
            0x8B08_0108,
            0x9100_2277,
            0xEB17_011F,
            0x9A97_8117,
            0x9100_06E0,
            0xD280_0101,
        ],
        "calloc",
        out,
    )?;
    let fail_calloc = arm64_fwd(0xB400_0000, out);
    let no_copy = arm64_jcc(&[0xF900_0017, 0xAA00_03F7], 0xB400_0016, out);
    c.words_call(
        &[0x9100_22E0, 0x9100_22C1, 0xF940_02C2, 0xD37D_F042],
        "memcpy",
        out,
    )?;
    c.words_call(&[0xAA16_03E0], "free", out)?;
    arm64_here(&[no_copy], out);
    c.words_call(
        &[
            0xAA17_03F6,
            if c.win { 0xB940_0AA0 } else { 0xF940_06A0 },
            0xAA16_03E1,
        ],
        c.pick("TlsSetValue", "pthread_setspecific"),
        out,
    )?;
    let fail_set = arm64_fwd(if c.win { 0x3400_0000 } else { 0x3500_0000 }, out);
    arm64_here(&[cell_skip], out);
    let fail_align = if c.win {
        c.words_call(&[0xAA14_03E0, 0xD280_0801], "_aligned_malloc", out)?;
        arm64_fwd(0xB400_0000, out)
    } else {
        c.words_call(
            &[0xF900_23FF, 0x9101_03E0, 0xD280_0801, 0xAA14_03E2],
            "posix_memalign",
            out,
        )?;
        let f = arm64_fwd(0x3500_0000, out);
        out.u32(0xF940_23E0);
        f
    };
    c.words_call(&[0x5280_0001, 0xAA14_03E2], "memset", out)?;
    out.words(&[0x8B13_0EC8, 0xF900_0500]);
    let skip_init = arm64_fwd(0xB400_0018, out);
    out.words(&[0xF940_0308, 0xAA00_03F8, 0xD63F_0100, 0xAA18_03E0]);
    arm64_here(&[skip_init], out);
    let done_ok = arm64_fwd(0x1400_0000, out);
    arm64_here(&[fail_calloc, fail_set, fail_align], out);
    out.u32(0xD280_0000);
    arm64_here(&[hit, done_ok], out);
    out.words(&[
        0xA943_63F7,
        0xA942_5BF5,
        0xA941_53F3,
        0xA940_7BFD,
        0x9101_43FF,
        0xD65F_03C0,
    ]);
    Ok(())
}

const ALLOC_CACHE_MAX_M1: u32 = haste_codegen::FRAME_CACHE_LIMIT - 17;

fn emit_arm64_frame_base(c: &Emit<'_>, addrs: SectionAddrs, out: &mut Buf) {
    tls::emit_arm64_tlv_base(c, addrs.tlv_desc, out);
    if addrs.frame_tls != 0 {
        out.u32(0x9100_0000 | ((addrs.frame_tls as u32 & 0xFFF) << 10));
    }
}

fn emit_arm64_dealloc(c: &Emit<'_>, addrs: SectionAddrs, out: &mut Buf) -> Result<()> {
    if c.win || addrs.tlv_desc == 0 {
        return c.tail(c.pick("_aligned_free", "free"), out);
    }
    let over_align = arm64_jcc(&[0xF100_405F], 0x5400_0008, out);
    let big = arm64_jcc(
        &[0xD100_0429, 0xF100_013F | (ALLOC_CACHE_MAX_M1 << 10)],
        0x5400_0008,
        out,
    );
    out.words(&[0x9100_3C2B, 0x927C_ED6B, 0xAA00_03E9, 0xAA1E_03EA]);
    emit_arm64_frame_base(c, addrs, out);
    let full = arm64_jcc(
        &[
            0x8B0B_0002,
            0xF940_0448,
            0xF100_011F | (u32::from(haste_codegen::FRAME_CACHE_DEPTH) << 10),
        ],
        0x5400_0002,
        out,
    );
    out.words(&[
        0xF940_004B,
        0xF900_012B,
        0xF900_0049,
        0x9100_0508,
        0xF900_0448,
        0xAA0A_03FE,
        0xD65F_03C0,
    ]);
    arm64_here(&[full], out);
    out.words(&[0xAA09_03E0, 0xAA0A_03FE]);
    arm64_here(&[over_align, big], out);
    c.tail("free", out)
}

fn emit_arm64_alloc(c: &Emit<'_>, addrs: SectionAddrs, out: &mut Buf) -> Result<()> {
    if c.win {
        return c.words_tail(
            &[
                0xF100_001F,
                0xD280_0029,
                0x9A89_1000,
                0xF100_003F,
                0x9A89_1021,
            ],
            "_aligned_malloc",
            out,
        );
    }
    if addrs.tlv_desc != 0 {
        let over_align = arm64_jcc(&[0xF100_403F], 0x5400_0008, out);
        let big = arm64_jcc(
            &[0xD100_0409, 0xF100_013F | (ALLOC_CACHE_MAX_M1 << 10)],
            0x5400_0008,
            out,
        );
        out.words(&[0x9100_3C09, 0x927C_ED29, 0xAA1E_03EA]);
        emit_arm64_frame_base(c, addrs, out);
        let miss = arm64_jcc(&[0x8B09_0002, 0xF940_0040], 0xB400_0000, out);
        out.words(&[
            0xF940_0008,
            0xF900_0048,
            0xF940_0449,
            0xD100_0529,
            0xF900_0449,
            0xAA0A_03FE,
            0xD65F_03C0,
        ]);
        arm64_here(&[miss], out);
        c.words_tail(&[0xAA09_03E0, 0xAA0A_03FE], "malloc", out)?;
        arm64_here(&[big], out);
        c.words_tail(&[0xF100_001F, 0xD280_0029, 0x9A89_1000], "malloc", out)?;
        arm64_here(&[over_align], out);
        out.words(&[0xF100_001F, 0xD280_0029, 0x9A89_1000]);
    } else {
        c.words_tail(
            &[
                0xF100_001F,
                0xD280_0029,
                0x9A89_1000,
                0xF100_403F,
                0x5400_0048,
            ],
            "malloc",
            out,
        )?;
    }
    c.words_call(
        &[
            0xD100_83FF,
            0xA900_7BFD,
            0xAA00_03E2,
            0xF900_0BFF,
            0x9100_43E0,
        ],
        "posix_memalign",
        out,
    )?;
    out.words(&[
        0x7100_001F,
        0xF940_0BE0,
        0x9A9F_0000,
        0xA8C2_7BFD,
        0xD65F_03C0,
    ]);
    Ok(())
}

fn emit_arm64_alloc_zeroed(c: &Emit<'_>, addrs: SectionAddrs, out: &mut Buf) -> Result<()> {
    if c.win {
        c.words_call(
            &[
                0xA9BE_7BFD,
                0xF900_0BF3,
                0xAA00_03F3,
                0xF100_001F,
                0xD280_0029,
                0x9A89_1000,
                0xF100_003F,
                0x9A89_1021,
            ],
            "_aligned_malloc",
            out,
        )?;
        let done = arm64_fwd(0xB400_0000, out);
        c.words_call(&[0x5280_0001, 0xAA13_03E2], "memset", out)?;
        arm64_patch(done, out.pos(), out);
        out.words(&[0xF940_0BF3, 0xA8C2_7BFD, 0xD65F_03C0]);
        return Ok(());
    }
    if addrs.tlv_desc != 0 {
        let over_align = arm64_jcc(&[0xF100_403F], 0x5400_0008, out);
        let big = arm64_jcc(
            &[0xD100_0409, 0xF100_013F | (ALLOC_CACHE_MAX_M1 << 10)],
            0x5400_0008,
            out,
        );
        out.words(&[0x9100_3C09, 0x927C_ED29, 0xAA1E_03EA]);
        emit_arm64_frame_base(c, addrs, out);
        let miss = arm64_jcc(&[0x8B09_0002, 0xF940_0040], 0xB400_0000, out);
        out.words(&[
            0xF940_0008,
            0xF900_0048,
            0xF940_0448,
            0xD100_0508,
            0xF900_0448,
            0xAA00_03EB,
        ]);
        let zero_loop = out.pos();
        out.words(&[0xA881_7D7F, 0xF100_4129]);
        let back = arm64_fwd(0x5400_0001, out);
        arm64_patch(back, zero_loop, out);
        out.words(&[0xAA0A_03FE, 0xD65F_03C0]);
        arm64_here(&[miss], out);
        c.words_tail(&[0xAA09_03E1, 0xD280_0020, 0xAA0A_03FE], "calloc", out)?;
        arm64_here(&[big], out);
        c.words_tail(
            &[
                0xF100_001F,
                0xD280_0029,
                0x9A89_1000,
                0xAA00_03E1,
                0xD280_0020,
            ],
            "calloc",
            out,
        )?;
        arm64_here(&[over_align], out);
        out.words(&[0xF100_001F, 0xD280_0029, 0x9A89_1000]);
    } else {
        c.words_tail(
            &[
                0xF100_001F,
                0xD280_0029,
                0x9A89_1000,
                0xF100_403F,
                0x5400_0088,
                0xAA00_03E1,
                0xD280_0020,
            ],
            "calloc",
            out,
        )?;
    }
    c.words_call(
        &[
            0xD100_83FF,
            0xA900_7BFD,
            0xF900_0BF3,
            0xAA00_03F3,
            0xAA00_03E2,
            0xF900_0FFF,
            0x9100_63E0,
        ],
        "posix_memalign",
        out,
    )?;
    c.words_call(
        &[0x3500_00C0, 0xF940_0FE0, 0x5280_0001, 0xAA13_03E2],
        "memset",
        out,
    )?;
    out.words(&[
        0x1400_0002,
        0xD280_0000,
        0xF940_0BF3,
        0xA8C2_7BFD,
        0xD65F_03C0,
    ]);
    Ok(())
}

fn emit_arm64_realloc(c: &Emit<'_>, addrs: SectionAddrs, out: &mut Buf) -> Result<()> {
    if c.win {
        return c.words_tail(
            &[
                0xF100_007F,
                0xD280_0029,
                0x9A89_1061,
                0xF100_005F,
                0x9A89_1042,
            ],
            "_aligned_realloc",
            out,
        );
    }
    if addrs.tlv_desc != 0 {
        let over_align = arm64_jcc(&[0xF100_405F], 0x5400_0008, out);
        let big_new = arm64_jcc(
            &[0xD100_0469, 0xF100_013F | (ALLOC_CACHE_MAX_M1 << 10)],
            0x5400_0008,
            out,
        );
        out.words(&[0x9100_3C63, 0x927C_EC63]);
        let big_old = arm64_jcc(
            &[0xD100_0429, 0xF100_013F | (ALLOC_CACHE_MAX_M1 << 10)],
            0x5400_0008,
            out,
        );
        let diff_class = arm64_jcc(&[0x9100_3C29, 0x927C_ED29, 0xEB03_013F], 0x5400_0001, out);
        out.u32(0xD65F_03C0);
        arm64_here(&[big_new], out);
        out.words(&[0xF100_007F, 0xD280_0029, 0x9A89_1063]);
        arm64_here(&[big_old, diff_class], out);
        c.words_tail(&[0xAA03_03E1], "realloc", out)?;
        arm64_here(&[over_align], out);
        out.words(&[0xF100_007F, 0xD280_0029, 0x9A89_1063]);
    } else {
        c.words_tail(
            &[
                0xF100_007F,
                0xD280_0029,
                0x9A89_1063,
                0xF100_405F,
                0x5400_0068,
                0xAA03_03E1,
            ],
            "realloc",
            out,
        )?;
    }
    c.words_call(
        &[
            0xD100_C3FF,
            0xA900_7BFD,
            0xA901_53F3,
            0xF900_13F5,
            0xAA00_03F3,
            0xEB03_003F,
            0x9A83_3034,
            0xAA02_03E1,
            0xAA03_03E2,
            0xF900_17FF,
            0x9100_A3E0,
        ],
        "posix_memalign",
        out,
    )?;
    c.words_call(
        &[
            0x3500_0140,
            0xF940_17F5,
            0xAA15_03E0,
            0xAA13_03E1,
            0xAA14_03E2,
        ],
        "memcpy",
        out,
    )?;
    c.words_call(&[0xAA13_03E0], "free", out)?;
    out.words(&[
        0xAA15_03E0,
        0x1400_0002,
        0xD280_0000,
        0xA941_53F3,
        0xF940_13F5,
        0xA8C3_7BFD,
        0xD65F_03C0,
    ]);
    Ok(())
}

fn emit_arm64_fill_once(c: &Emit<'_>, os: Os, out: &mut Buf) -> Result<()> {
    out.words(&[
        0xA9BD_7BFD,
        0xA901_53F3,
        0xF900_13F5,
        0xAA00_03F3,
        0xAA01_03F4,
    ]);
    match os {
        Os::Darwin => out.words(&[0xD53B_D075, 0x927D_F2B5]),
        Os::Windows => out.u32(0xAA12_03F5),
        Os::Linux => out.u32(0xD53B_D055),
    }
    let retry = out.pos();
    c.words_call(
        &[
            0xC85F_FE68,
            0xB500_00E8,
            0xC809_FE75,
            0x35FF_FFA9,
            0xD63F_0280,
            0xD280_0028,
            0xC89F_FE68,
            0x1400_0007,
            0xF100_051F,
            0x5400_00A0,
            0xEB15_011F,
            0x5400_0060,
        ],
        c.pick("SwitchToThread", "sched_yield"),
        out,
    )?;
    let back = ((retry as isize - out.pos() as isize) >> 2) as u32;
    out.words(&[
        0x1400_0000 | (back & 0x03FF_FFFF),
        0xF940_13F5,
        0xA941_53F3,
        0xA8C3_7BFD,
        0xD65F_03C0,
    ]);
    Ok(())
}

pub(super) fn emit_shim_arm64(
    os: Os,
    shim: Shim,
    va: u64,
    addrs: SectionAddrs,
    stub_va: &dyn Fn(&str) -> Result<u64>,
    out: &mut Buf,
) -> Result<()> {
    let c = Emit {
        arch: Arch::Arm64,
        win: os == Os::Windows,
        va,
        base: out.pos(),
        stub_va,
    };
    match shim {
        Shim::Noop => out.u32(0xD65F_03C0),
        Shim::Barrier => out.words(&[0xD503_3F9F, 0xD503_3FDF, 0xD65F_03C0]),
        Shim::Syscall => out.words(&[
            if os == Os::Darwin {
                0x9280_09A0
            } else {
                0x9280_04A0
            },
            0xD65F_03C0,
        ]),
        Shim::Cpuid => out.words(&[0xA900_7C1F, 0xD65F_03C0]),
        Shim::ExtendedState => out.words(&[0xD280_0000, 0xD65F_03C0]),
        Shim::Dealloc => emit_arm64_dealloc(&c, addrs, out)?,
        Shim::FrameAlloc | Shim::FrameFree if addrs.tlv_desc != 0 => {
            tls::emit_arm64_tlv_frame_cache(
                &c,
                shim == Shim::FrameAlloc,
                addrs.tlv_desc,
                addrs.frame_tls,
                out,
            )?
        }
        Shim::FrameAlloc | Shim::FrameFree => c.tail(
            if shim == Shim::FrameAlloc {
                "malloc"
            } else {
                "free"
            },
            out,
        )?,
        Shim::Alloc => emit_arm64_alloc(&c, addrs, out)?,
        Shim::AllocZeroed => emit_arm64_alloc_zeroed(&c, addrs, out)?,
        Shim::Realloc => emit_arm64_realloc(&c, addrs, out)?,
        Shim::FillOnce => emit_arm64_fill_once(&c, os, out)?,
        Shim::ThreadCell if addrs.tlv_desc != 0 => {
            tls::emit_arm64_tlv_thread_cell(&c, addrs.tlv_desc, addrs.thread_tls, out)?
        }
        Shim::ThreadCell => emit_arm64_thread_cell(&c, addrs.thread_key, out)?,
        Shim::CatchUnwind => unwind::emit_arm64_catch_unwind(&c, addrs, out)?,
        Shim::StartPanic => unwind::emit_arm64_start_panic(&c, addrs, out)?,
        Shim::StartPanicAbort => c.tail("abort", out)?,
        Shim::PanicCleanup => out.words(&[0xA940_0400, 0xD65F_03C0]),
        Shim::DropPush | Shim::DropPop => {
            emit_arm64_drop_op(&c, shim == Shim::DropPush, addrs, out)?
        }
        Shim::UnwindWalk => unwind::emit_walk(Arch::Arm64, os, out)?,
        Shim::Wide(wide) => wide::emit(Arch::Arm64, os, wide, out)?,
    }
    Ok(())
}
