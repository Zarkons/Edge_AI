#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

void kernel_main()
{
    DPRINT("My logical coordinates are {},{}\n", get_absolute_logical_x(), get_absolute_logical_y());
    uint32_t dst_base_addr = get_arg_val<uint32_t>(0);
    uint32_t page_idx = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;
    const uint32_t tile_size_bytes = get_tile_size(cb_out0);

    // Compute direct byte offset destination
    uint32_t dst_offset_addr = dst_base_addr + (page_idx * tile_size_bytes);
    uint32_t dram_bank = 0;

    cb_wait_front(cb_out0, 1);
    uint32_t cb_out0_addr = get_read_ptr(cb_out0);

    // Convert address explicitly to bypass Interleaved alignment traps
    unsigned long long noc_dst_addr = get_noc_addr_from_bank_id<true>(dram_bank, dst_offset_addr);

    noc_async_write(cb_out0_addr, noc_dst_addr, tile_size_bytes);
    noc_async_write_barrier();

    cb_pop_front(cb_out0, 1);
}
