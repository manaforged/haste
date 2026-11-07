use std::sync::Mutex;

const FRAMES: usize = 24;
const SIGNALS: [i32; 5] = [libc::SIGABRT, libc::SIGSEGV, libc::SIGBUS, libc::SIGILL, libc::SIGTRAP];
const STACK_BYTES: usize = 1 << 18;

/// A call site that has an exception handler, as offsets into its function.
#[derive(Clone, Copy)]
pub struct Site {
    pub ret: u32,
    pub handler: u32,
    pub frame_offset: u32,
}

struct Code {
    start: usize,
    size: usize,
    name: String,
    sites: Vec<Site>,
}

/// What owns a return address.
pub enum Frame {
    Handler { pc: usize, frame_offset: usize },
    Generated,
    Native(String),
}

static CODE: Mutex<Vec<Code>> = Mutex::new(Vec::new());

pub fn remember(start: usize, size: usize, name: String, sites: Vec<Site>) {
    if let Ok(mut code) = CODE.lock() {
        code.push(Code { start, size, name, sites });
    }
}

pub fn frame_at(ret: usize) -> Frame {
    let code = CODE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(item) = code.iter().rev().find(|item| ret >= item.start && ret < item.start + item.size) else {
        return Frame::Native(host_name(ret).unwrap_or_else(|| format!("{ret:#x}")));
    };
    let offset = ret - item.start;
    match item.sites.iter().find(|site| site.ret as usize == offset) {
        Some(site) => Frame::Handler { pc: item.start + site.handler as usize, frame_offset: site.frame_offset as usize },
        None => Frame::Generated,
    }
}

fn name_at(code: &[Code], pc: usize) -> Option<&str> {
    code.iter().rev().find(|item| pc >= item.start && pc < item.start + item.size).map(|item| item.name.as_str())
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe fn registers(context: *mut libc::c_void) -> (usize, usize) {
    // SAFETY: context is the ucontext the kernel passed to an SA_SIGINFO handler.
    let state = unsafe { (*(*context.cast::<libc::ucontext_t>()).uc_mcontext).__ss };
    (state.__pc as usize, state.__fp as usize)
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
unsafe fn registers(context: *mut libc::c_void) -> (usize, usize) {
    // SAFETY: context is the ucontext the kernel passed to an SA_SIGINFO handler.
    let state = unsafe { &(*context.cast::<libc::ucontext_t>()).uc_mcontext };
    (state.gregs[libc::REG_RIP as usize] as usize, state.gregs[libc::REG_RBP as usize] as usize)
}

fn host_name(pc: usize) -> Option<String> {
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    // SAFETY: dladdr only reads pc as a number and fills the local info.
    if unsafe { libc::dladdr(pc as *const libc::c_void, &mut info) } == 0 {
        return None;
    }
    let text = if info.dli_sname.is_null() { info.dli_fname } else { info.dli_sname };
    // SAFETY: dladdr filled text with a nul-terminated name that lives as long as the image, or left it null.
    (!text.is_null()).then(|| unsafe { std::ffi::CStr::from_ptr(text) }.to_string_lossy().into_owned())
}

extern "C" fn on_fault(signal: i32, info: *mut libc::siginfo_t, context: *mut libc::c_void) {
    // SAFETY: the kernel passes a valid siginfo and ucontext to an SA_SIGINFO handler.
    let (fault, (start_pc, start_frame)) = unsafe { ((*info).si_addr() as usize, registers(context)) };
    eprintln!("fault: signal {signal} address {fault:#x}");
    if let Ok(code) = CODE.try_lock() {
        let (mut pc, mut frame) = (start_pc, start_frame);
        for _ in 0..FRAMES {
            if let Some(name) = name_at(&code, pc) {
                eprintln!("  at {name}");
            } else if let Some(name) = host_name(pc) {
                eprintln!("  in {name}");
            }
            if frame == 0 || frame % 8 != 0 {
                break;
            }
            // SAFETY: generated code and the host keep frame pointers, so frame points at a saved (frame, return address) pair; a bad walk ends in the default handler below.
            let (next, ret) = unsafe { (*(frame as *const usize), *((frame + 8) as *const usize)) };
            if next <= frame {
                break;
            }
            (pc, frame) = (ret, next);
        }
    }
    // SAFETY: restoring the default action and returning re-raises the fault with its normal exit status.
    unsafe { libc::signal(signal, libc::SIG_DFL) };
}

pub fn install() {
    let stack = Box::leak(vec![0_u8; STACK_BYTES].into_boxed_slice());
    let alternate = libc::stack_t { ss_sp: stack.as_mut_ptr().cast(), ss_flags: 0, ss_size: STACK_BYTES };
    // SAFETY: the alternate stack is leaked, and every pointer passed to the kernel is valid for the call.
    unsafe {
        libc::sigaltstack(&alternate, std::ptr::null_mut());
        for signal in SIGNALS {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = on_fault as *const () as usize;
            action.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
            libc::sigaction(signal, &action, std::ptr::null_mut());
        }
    }
}
