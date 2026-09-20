use std::arch::global_asm;

global_asm!(
    r#"
    .text

    .global nexium_nce_enter_trampoline
    .type nexium_nce_enter_trampoline, %function
    .p2align 2
nexium_nce_enter_trampoline:
    mov     x3, sp
    mrs     x4, tpidr_el0
    ldr     x5, [x1, #0xF8]
    mov     sp, x5
    add     x5, x1, #0x320
    stp     x3, x4, [x5, #0xE0]
    stp     x19, x20, [x5, #0x00]
    stp     x21, x22, [x5, #0x10]
    stp     x23, x24, [x5, #0x20]
    stp     x25, x26, [x5, #0x30]
    stp     x27, x28, [x5, #0x40]
    stp     x29, x30, [x5, #0x50]
    stp     q8, q9,   [x5, #0x60]
    stp     q10, q11, [x5, #0x80]
    stp     q12, q13, [x5, #0xA0]
    stp     q14, q15, [x5, #0xC0]
    msr     tpidr_el0, x0
    br      x2
    .size nexium_nce_enter_trampoline, .-nexium_nce_enter_trampoline

    .global nexium_nce_enter_signal
    .type nexium_nce_enter_signal, %function
    .p2align 2
nexium_nce_enter_signal:
    mov     x9, x1
    mov     x8, #130
    mov     x1, #12
    svc     #0
    brk     #1000
    .size nexium_nce_enter_signal, .-nexium_nce_enter_signal

    .global nexium_nce_sigusr2_handler
    .type nexium_nce_sigusr2_handler, %function
    .p2align 2
nexium_nce_sigusr2_handler:
    stp     x29, x30, [sp, #-0x10]!
    mov     x29, sp
    mov     x0, x2
    bl      nexium_nce_restore_guest_context
    mrs     x8, tpidr_el0
    ldr     x9, [x0, #0x10]
    str     x8, [x9, #0x408]
    msr     tpidr_el0, x0
    bl      nexium_nce_unlock_nep
    ldp     x29, x30, [sp], #0x10
    ret
    .size nexium_nce_sigusr2_handler, .-nexium_nce_sigusr2_handler

    .global nexium_nce_sigurg_handler
    .type nexium_nce_sigurg_handler, %function
    .p2align 2
nexium_nce_sigurg_handler:
    mrs     x8, tpidr_el0
    ldr     w9, [x8, #0x20]
    mov     w10, #0x5855
    movk    w10, #0x4E45, lsl #16
    cmp     w9, w10
    b.ne    1f
    ldr     w9, [x8, #0x18]
    cbz     w9, 1f
    ldr     x0, [x8, #0x10]
    ldr     x3, [x0, #0x408]
    msr     tpidr_el0, x3
    mov     x1, x2
    b       nexium_nce_save_guest_context
1:
    ret
    .size nexium_nce_sigurg_handler, .-nexium_nce_sigurg_handler

    .macro NEXIUM_NCE_FAULT_HANDLER name, guest_fn, host_fn
    .global \name
    .type \name, %function
    .p2align 2
\name:
    mrs     x8, tpidr_el0
    ldr     w9, [x8, #0x20]
    mov     w10, #0x5855
    movk    w10, #0x4E45, lsl #16
    cmp     w9, w10
    b.eq    1f
    b       \host_fn
1:
    stp     x29, x30, [sp, #-0x20]!
    str     x19, [sp, #0x10]
    mov     x29, sp
    mov     x19, x8
    ldr     x0, [x8, #0x10]
    ldr     x3, [x0, #0x408]
    msr     tpidr_el0, x3
    bl      \guest_fn
    cbz     x0, 2f
    msr     tpidr_el0, x19
2:
    ldr     x19, [sp, #0x10]
    ldp     x29, x30, [sp], #0x20
    ret
    .size \name, .-\name
    .endm

    NEXIUM_NCE_FAULT_HANDLER nexium_nce_sigsegv_handler, nexium_nce_guest_segv, nexium_nce_host_segv
    NEXIUM_NCE_FAULT_HANDLER nexium_nce_sigbus_handler, nexium_nce_guest_bus, nexium_nce_host_bus
    NEXIUM_NCE_FAULT_HANDLER nexium_nce_sigtrap_handler, nexium_nce_guest_trap, nexium_nce_host_trap
    NEXIUM_NCE_FAULT_HANDLER nexium_nce_sigill_handler, nexium_nce_guest_ill, nexium_nce_host_ill
    "#
);

extern "C" {
    pub fn nexium_nce_enter_trampoline(nep: *mut u8, ctx: *mut u8, trampoline: u64) -> u64;
    pub fn nexium_nce_enter_signal(tid: i32, nep: *mut u8) -> u64;
    pub fn nexium_nce_sigusr2_handler(sig: i32, info: *mut libc::siginfo_t, ctx: *mut libc::c_void);
    pub fn nexium_nce_sigurg_handler(sig: i32, info: *mut libc::siginfo_t, ctx: *mut libc::c_void);
    pub fn nexium_nce_sigsegv_handler(sig: i32, info: *mut libc::siginfo_t, ctx: *mut libc::c_void);
    pub fn nexium_nce_sigbus_handler(sig: i32, info: *mut libc::siginfo_t, ctx: *mut libc::c_void);
    pub fn nexium_nce_sigtrap_handler(sig: i32, info: *mut libc::siginfo_t, ctx: *mut libc::c_void);
    pub fn nexium_nce_sigill_handler(sig: i32, info: *mut libc::siginfo_t, ctx: *mut libc::c_void);
}
