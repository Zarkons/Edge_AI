#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

void kernel_main()
{
    DPRINT("My logical coordinates are {},{}\n", get_absolute_logical_x(), get_absolute_logical_y());
    uint32_t in0_base_addr = get_arg_val<uint32_t>(0);
    uint32_t in1_base_addr = get_arg_val<uint32_t>(1);
    uint32_t page_idx = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    const uint32_t tile_size_bytes = get_tile_size(cb_in0);

    // 2. Compute exact aligned byte offset for this core's assigned tile
    uint32_t src0_offset_addr = in0_base_addr + (page_idx * tile_size_bytes);
    uint32_t src1_offset_addr = in1_base_addr + (page_idx * tile_size_bytes);

    cb_reserve_back(cb_in0, 1);
    cb_reserve_back(cb_in1, 1);

    uint32_t cb_in0_addr = get_write_ptr(cb_in0);
    uint32_t cb_in1_addr = get_write_ptr(cb_in1);

    // 3. Use standard NoC async read with an explicit coordinate tracker
    // Replicated DRAM buffers default to bank 0 on target channels
    uint32_t dram_bank = 0;

    // Simple, explicitly aligned direct read bypassing interleaved page calculation:
    unsigned long long noc_src0_addr = get_noc_addr_from_bank_id<true>(dram_bank, src0_offset_addr);
    unsigned long long noc_src1_addr = get_noc_addr_from_bank_id<true>(dram_bank, src1_offset_addr);

    noc_async_read(noc_src0_addr, cb_in0_addr, tile_size_bytes);
    noc_async_read(noc_src1_addr, cb_in1_addr, tile_size_bytes);

    noc_async_read_barrier();

    cb_push_back(cb_in0, 1);
    cb_push_back(cb_in1, 1);
}
