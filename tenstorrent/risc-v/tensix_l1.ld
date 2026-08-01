/* risc-v/tensix_l1.ld */

ENTRY(_start) /* Tells the linker that boot.S goes first */

MEMORY
{
    /* Slide origins to start at the 0x80000000 boundary */
    IMEM_CODE (rx)  : ORIGIN = 0x80000000, LENGTH = 64K   
    DMEM_SRAM (rw)  : ORIGIN = 0x80010000, LENGTH = 960K  
}

SECTIONS
{
    .text : {
        _text_start = .;
        *(.text._start) /* Forces the custom bootstrapper to sit at 0x80000000 */
        *(.text .text.*)
        _text_end = .;
    } > IMEM_CODE

    .rodata : { *(.rodata .rodata.*) } > DMEM_SRAM
    .data   : { *(.data .data.*) } > DMEM_SRAM
    .bss    : {
        _bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        _bss_end = .;
    } > DMEM_SRAM
    .htif : ALIGN(8) {
        PROVIDE(tohost = .);
        KEEP(*(.htif))
        . = . + 8;
        PROVIDE(fromhost = .);
        KEEP(*(.htif))
        . = . + 8;
    } > DMEM_SRAM


    _stack_top = ORIGIN(DMEM_SRAM) + LENGTH(DMEM_SRAM);
}
