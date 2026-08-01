#include <iostream>
#include <vector>
#include <memory>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;

int main()
{
    // 1. Initialize the hardware mesh and grab the command queue
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    distributed::MeshCommandQueue &cq = mesh_device->mesh_command_queue();

    // 2. Prepare the workload and core ranges
    distributed::MeshWorkload workload;
    Program program = CreateProgram();

    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {1, 1};

    // FIX: Explicitly namespace CoreRange to eliminate the deprecation warning
    tt::tt_metal::CoreRange mesh_range = {start_core, end_core};

    // Explicitly namespaced CoreRangeSet
    tt::tt_metal::CoreRangeSet multi_core_mesh({mesh_range});

    // Define data sizes (1 Tile = 32x32 elements. Float16_b = 2 bytes per element)
    uint32_t tile_size_bytes = 32 * 32 * 2;
    uint32_t total_buffer_size = 4 * tile_size_bytes; // 4 tiles total for the 2x2 grid

    // 3. Configure and Allocate Distributed Mesh Buffers in DRAM
    distributed::MeshBufferConfig mesh_buffer_config = distributed::ReplicatedBufferConfig{
        .size = total_buffer_size};

    distributed::DeviceLocalBufferConfig local_config{
        .page_size = tile_size_bytes,
        .buffer_type = BufferType::DRAM};

    std::shared_ptr<distributed::MeshBuffer> src0_dram_buffer = distributed::MeshBuffer::create(mesh_buffer_config, local_config, mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> src1_dram_buffer = distributed::MeshBuffer::create(mesh_buffer_config, local_config, mesh_device.get());
    std::shared_ptr<distributed::MeshBuffer> dst_dram_buffer = distributed::MeshBuffer::create(mesh_buffer_config, local_config, mesh_device.get());

    // 4. Configure Local L1 Circular Buffers (CBs) on the Tensix Core
    CircularBufferConfig cb_in0_config = CircularBufferConfig(2 * tile_size_bytes, {{tt::CBIndex::c_0, DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_0, tile_size_bytes);
    CreateCircularBuffer(program, multi_core_mesh, cb_in0_config);

    CircularBufferConfig cb_in1_config = CircularBufferConfig(2 * tile_size_bytes, {{tt::CBIndex::c_1, DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_1, tile_size_bytes);
    CreateCircularBuffer(program, multi_core_mesh, cb_in1_config);

    CircularBufferConfig cb_out0_config = CircularBufferConfig(2 * tile_size_bytes, {{tt::CBIndex::c_16, DataFormat::Float16_b}}).set_page_size(tt::CBIndex::c_16, tile_size_bytes);
    CreateCircularBuffer(program, multi_core_mesh, cb_out0_config);

    // 5. Compile and Attach Kernels to the Program Range
    KernelHandle reader_kernel_id = CreateKernel(
        program, "kernels/NCRISC.cpp", multi_core_mesh,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle writer_kernel_id = CreateKernel(
        program, "kernels/BRISC.cpp", multi_core_mesh,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    KernelHandle compute_kernel_id = CreateKernel(
        program, "kernels/TRISC.cpp", multi_core_mesh,
        ComputeConfig{.math_fidelity = tt::tt_metal::MathFidelity::HiFi4, .fp32_dest_acc_en = false, .math_approx_mode = false});

    // 6. Set Runtime Arguments for Dataflow Kernels across the 2x2 Grid
    uint32_t src0_addr = static_cast<uint32_t>(src0_dram_buffer->address());
    uint32_t src1_addr = static_cast<uint32_t>(src1_dram_buffer->address());
    uint32_t dst_addr = static_cast<uint32_t>(dst_dram_buffer->address());

    uint32_t page_index = 0;
    for (uint32_t y = start_core.y; y <= end_core.y; y++)
    {
        for (uint32_t x = start_core.x; x <= end_core.x; x++)
        {
            CoreCoord target_core = {x, y};

            SetRuntimeArgs(program, reader_kernel_id, target_core, {
                                                                       src0_addr, // get_arg_val(0)
                                                                       src1_addr, // get_arg_val(1)
                                                                       page_index // get_arg_val(2)
                                                                   });

            SetRuntimeArgs(program, writer_kernel_id, target_core, {
                                                                       dst_addr,  // get_arg_val(0)
                                                                       page_index // get_arg_val(1)
                                                                   });

            page_index++;
        }
    }

    // 7. Prepare Host Data (Packed as uint32_t to align with hardware word sizes)
    uint32_t elements_count = total_buffer_size / sizeof(uint32_t);
    std::vector<uint32_t> host_src0(elements_count, 0x3C003C00); // Packed Float16 representation of 1.0
    std::vector<uint32_t> host_src1(elements_count, 0x40004000); // Packed Float16 representation of 2.0
    std::vector<uint32_t> host_dst(elements_count, 0);

    // 8. Execute via the Distributed Mesh Queue API
    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, host_src0, /*blocking=*/false);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, host_src1, /*blocking=*/false);

    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

    distributed::Finish(cq);

    distributed::EnqueueReadMeshBuffer(cq, host_dst, dst_dram_buffer, /*blocking=*/true);

    // 9. Output verification, unpacking, and dual-format printing
    std::cout << "\n================= VERIFICATION =================\n";
    std::cout << "Expected Output Value Per 16-bit Float Slot:\n";
    std::cout << "  -> Hex: 0x4200\n";
    std::cout << "  -> Dec: 3.0\n";
    std::cout << "Expected Packed 32-bit Word:\n";
    std::cout << "  -> Hex: 0x42004200\n";
    std::cout << "------------------------------------------------\n";

    bool all_passed = true;
    uint32_t expected_packed = 0x42004200; // Packed representation of [3.0, 3.0]

    // Print the first few returned elements for visual inspection
    uint32_t print_limit = 4;
    for (uint32_t i = 0; i < print_limit && i < host_dst.size(); i++)
    {
        uint32_t raw_word = host_dst[i];

        // Isolate the upper and lower 16-bit float components
        uint16_t upper_half = (raw_word >> 16) & 0xFFFF;
        uint16_t lower_half = raw_word & 0xFFFF;

        std::cout << "Word [" << i << "] Actual Result:\n";
        std::cout << "  -> Full 32-bit Hex: 0x" << std::hex << raw_word << std::dec << "\n";

        // Match upper half
        std::cout << "  -> Upper 16-bit Float: Hex = 0x" << std::hex << upper_half << std::dec;
        if (upper_half == 0x4200)
            std::cout << " | Dec = 3.0 [PASS]\n";
        else
            std::cout << " | Dec = (Mismatch) [FAIL]\n";

        // Match lower half
        std::cout << "  -> Lower 16-bit Float: Hex = 0x" << std::hex << lower_half << std::dec;
        if (lower_half == 0x4200)
            std::cout << " | Dec = 3.0 [PASS]\n";
        else
            std::cout << " | Dec = (Mismatch) [FAIL]\n";
        std::cout << "\n";
    }

    // Verify every single element in the entire 4-tile buffer
    for (uint32_t val : host_dst)
    {
        if (val != expected_packed)
        {
            all_passed = false;
            break;
        }
    }

    std::cout << "------------------------------------------------\n";
    if (all_passed)
    {
        std::cout << "🚀 ALL TESTS PASSED! Hardware mesh computed 1.0 + 2.0 = 3.0 across all 4 tiles.\n";
    }
    else
    {
        std::cout << "❌ TEST FAILED! Output data mismatch detected.\n";
    }
    std::cout << "================================================\n\n";

    return 0;
}
