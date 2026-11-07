use std::sync::{Condvar, Mutex, PoisonError};
use std::time::Duration;

const MARK: u64 = 0x5041_524b_5345_4d41;
const FOREVER: u64 = u64::MAX;
const TIMED_OUT: isize = 49;

#[repr(C)]
pub struct Semaphore {
    mark: u64,
    count: Mutex<isize>,
    ready: Condvar,
}

#[repr(C)]
struct Timebase {
    numer: u32,
    denom: u32,
}

unsafe extern "C" {
    fn mach_absolute_time() -> u64;
    fn mach_timebase_info(info: *mut Timebase) -> i32;
}

fn remaining(deadline: u64) -> Duration {
    let mut base = Timebase { numer: 0, denom: 0 };
    // SAFETY: base is a live local of the layout the call fills, and the clock call takes nothing.
    let now = unsafe {
        mach_timebase_info(&mut base);
        mach_absolute_time()
    };
    let ticks = u128::from(deadline.saturating_sub(now));
    Duration::from_nanos((ticks * u128::from(base.numer) / u128::from(base.denom.max(1))) as u64)
}

/// Creates a semaphore that stays usable in a forked child.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_semaphore_create(value: isize) -> *mut Semaphore {
    Box::into_raw(Box::new(Semaphore { mark: MARK, count: Mutex::new(value), ready: Condvar::new() }))
}

/// Adds one to the semaphore and wakes a waiter.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_semaphore_signal(semaphore: *mut Semaphore) -> isize {
    // SAFETY: std passes a semaphore this file created and keeps it alive across the call.
    let semaphore = unsafe { &*semaphore };
    *semaphore.count.lock().unwrap_or_else(PoisonError::into_inner) += 1;
    semaphore.ready.notify_one();
    0
}

/// Takes one from the semaphore, waiting up to a deadline in mach ticks.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_semaphore_wait(semaphore: *mut Semaphore, deadline: u64) -> isize {
    // SAFETY: std passes a semaphore this file created and keeps it alive across the call.
    let semaphore = unsafe { &*semaphore };
    let mut count = semaphore.count.lock().unwrap_or_else(PoisonError::into_inner);
    while *count <= 0 {
        if deadline == FOREVER {
            count = semaphore.ready.wait(count).unwrap_or_else(PoisonError::into_inner);
            continue;
        }
        let left = remaining(deadline);
        if left.is_zero() {
            return TIMED_OUT;
        }
        count = semaphore.ready.wait_timeout(count, left).unwrap_or_else(PoisonError::into_inner).0;
    }
    *count -= 1;
    0
}

/// Frees a semaphore this file created.
#[unsafe(no_mangle)]
pub extern "C" fn dispatch_release(object: *mut Semaphore) {
    // SAFETY: every caller in this binary passes a live object, and an object from elsewhere starts with a pointer that never equals the mark.
    if unsafe { (*object).mark } == MARK {
        // SAFETY: the mark proves Box::into_raw made this pointer, and std releases a semaphore once.
        drop(unsafe { Box::from_raw(object) });
    }
}

/// Resolves generated code's semaphore calls to this file.
pub fn symbol(name: &str) -> Option<usize> {
    Some(match name {
        "dispatch_semaphore_create" => dispatch_semaphore_create as *const () as usize,
        "dispatch_semaphore_signal" => dispatch_semaphore_signal as *const () as usize,
        "dispatch_semaphore_wait" => dispatch_semaphore_wait as *const () as usize,
        "dispatch_release" => dispatch_release as *const () as usize,
        _ => return None,
    })
}
