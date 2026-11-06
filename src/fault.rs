use std::sync::Mutex;

const FRAMES: usize = 24;
const SIGNALS: [i32; 3] = [libc::SIGSEGV, libc::SIGBUS, libc::SIGILL];
const STACK_BYTES: usize = 1 << 18;

struct Code {
    start: usize,
    size: usize,
    name: String,
}

static CODE: Mutex<Vec<Code>> = Mutex::new(Vec::new());

pub fn remember(start: usize, size: usize, name: String) {
    if let Ok(mut code) = CODE.lock() {
        code.push(Code { start, size, name });
    }
}

fn name_at(code: &[Code], pc: usize) -> Option<&str> {
    code.iter().rev().find(|item| pc >= item.start && pc < item.start + item.size).map(|item| item.name.as_str())
}

extern "C" fn on_fault(signal: i32, info: *mut libc::siginfo_t, context: *mut libc::c_void) {
    // SAFETY: the kernel passes a valid siginfo and ucontext to an SA_SIGINFO handler.
    let (fault, state) = unsafe { ((*info).si_addr() as usize, (*(*context.cast::<libc::ucontext_t>()).uc_mcontext).__ss) };
    eprintln!("fault: signal {signal} address {fault:#x}");
    if let Ok(code) = CODE.try_lock() {
        let (mut pc, mut frame) = (state.__pc as usize, state.__fp as usize);
        for _ in 0..FRAMES {
            if let Some(name) = name_at(&code, pc) {
                eprintln!("  at {name}");
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
