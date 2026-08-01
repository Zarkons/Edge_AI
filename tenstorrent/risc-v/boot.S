# risc-v/boot.S
.section .text._start
.globl _start
.type _start, @function

_start:
    # Disable interrupts globally during hardware setup
    csrci mstatus, 8

    # CRITICAL: Enable the Floating-Point Unit (FPU) hardware
    # We load the bitmask 0x6000 (bits 13 and 14 of mstatus) into a temporary register
    # and use 'csrs' (CSR Set) to turn those hardware routing bits ON.
    li t0, 0x6000
    csrs mstatus, t0

    # Clear floating-point exception flags and set rounding mode to Nearest
    # 'fsrm' forces standard IEEE-754 rounding now that the unit is powered.
    csrwi fcsr, 0

    # Set up the Stack Pointer register (sp)
    # 'la' (Load Address) pulls the '_stack_top' constant calculation from the linker script.
    .option push
    .option norelax
    la sp, _stack_top
    .option pop

    # Securely execute compiled C++ main function loop
    # 'jal' (Jump and Link) jumps to 'main' and saves the return path to 'ra'.
    jal ra, main

_shutdown_host:
    # 1. Load the compiled address of the 'tohost' tracking control block
    la t0, tohost
    
    # 2. Write a status value of 1 (Success Exit Code) to the lowest 32 bits
    # Shift 1 left by 0 bits for the exit command parameter
    li t1, 1
    sw t1, 0(t0)

_backup_loop:
    # Fallback safety if the simulator doesn't instantly snap closed
    wfi
    j _backup_loop

.size _start, .-_start
