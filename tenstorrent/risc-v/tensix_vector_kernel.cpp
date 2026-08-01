// Simulated static L1 SRAM addresses typical inside a Tensix tile buffer
#define SRC_A_ADDR 0x80014000
#define SRC_B_ADDR 0x80014100
#define DST_OUT_ADDR 0x80014200

void execute_vector_multiply(int vector_size)
{
    volatile float *ptr_a = (float *)SRC_A_ADDR;
    volatile float *ptr_b = (float *)SRC_B_ADDR;
    volatile float *ptr_out = (float *)DST_OUT_ADDR;

    for (int i = 0; i < vector_size; i++)
    {
        // Direct inline assembly block utilizing the RISC-V F & D hardware float ISA extensions
        asm volatile(
            "flw f0, 0(%0)\n\t"     // Load Single-Precision float from pointer address %0 into register f0
            "flw f1, 0(%1)\n\t"     // Load Single-Precision float from pointer address %1 into register f1
            "fmul.s f2, f0, f1\n\t" // Perform high-speed floating-point hardware multiplication (f2 = f0 * f1)
            "fsw f2, 0(%2)\n\t"     // Store the computed output result back down to destination pointer %2
            :
            : "r"(ptr_a), "r"(ptr_b), "r"(ptr_out) // Input tokens %0, %1, %2 mapped to pointers
            : "f0", "f1", "f2", "memory"           // Clobber list warning the compiler we changed these registers
        );

        // Advance pointers to step to the next array data slot
        ptr_a++;
        ptr_b++;
        ptr_out++;
    }
}

int main()
{
    // Process a localized tile buffer chunk of 32 elements
    execute_vector_multiply(32);
    return 0;
}
