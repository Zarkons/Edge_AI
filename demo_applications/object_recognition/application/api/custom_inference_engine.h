#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>
#include "custom_kernels.h"
#include "inference_pipeline_structures.h"

class CustomInferenceEngine
{
public:
    CustomInferenceEngine(const std::vector<int64_t> &root_input_shape,
                          const std::string &root_input_name);

    ~CustomInferenceEngine()
    {
        if (arena_ptr_)
        {
            delete[] arena_ptr_;
            arena_ptr_ = nullptr;
        }
    }

    // Explicitly delete copy constructor and assignment operator to prevent pointer duplication bugs
    CustomInferenceEngine(const CustomInferenceEngine &) = delete;
    CustomInferenceEngine &operator=(const CustomInferenceEngine &) = delete;

    /**
     * @brief Analyzes input/output tensor life cycles, traces dynamic shapes,
     *        maps 16-byte aligned relative offsets, and sets up the execution plan.
     * @param blueprints Pre-parsed array of optimized architectural execution objects.
     * @return The minimum number of contiguous bytes the user must provide for the workspace.
     */
    size_t initialize_runtime_plan(std::vector<ExecutionStep> &blueprints);

    /**
     * @brief Executes full network pipeline over physical raw memory blocks with zero dynamic allocation.
     */
    void forward(const void *input_hardware_buffer,
                 void *output_hardware_buffer,
                 void *arena_workspace);

private:
    // --- Private Allocations and Registries ---

    uint8_t *arena_ptr_ = nullptr; // Flat continuous physical workspace allocation
    size_t arena_size_ = 0;

    std::vector<RuntimeNode> execution_plan_;
    std::unordered_map<std::string, WeightTensorView> weights_registry_;

    // Maps intermediate buffer strings to fixed byte offsets within the arena during model setup
    std::unordered_map<std::string, size_t> arena_offset_map_;
    CustomKernels kernels_;

    // Constructor-seeded root input shape for activation shape resolution.
    std::string root_input_name_;
    std::vector<int64_t> root_input_shape_;

    // --- Private Execution Kernel Routing ---

    /**
     * @brief Internal core router targeting isolated optimized element-wise and structural kernels.
     */
    void dispatch_kernel(const RuntimeNode &node);

    std::vector<int64_t> infer_output_shape_for_step(
        const ExecutionStep &step,
        const std::vector<std::vector<int64_t>> &resolved_input_shapes,
        const std::vector<std::string> &input_names,
        const std::unordered_map<std::string, std::vector<int64_t>> &shape_registry) const;
    std::vector<int64_t> infer_conv_output_shape(const ExecutionStep &step,
                                                 const std::vector<int64_t> &current_input_shape) const;
    std::vector<int64_t> infer_maxpool_output_shape(const ExecutionStep &step,
                                                    const std::vector<int64_t> &current_input_shape) const;
    std::vector<int64_t> infer_concat_output_shape(const ExecutionStep &step,
                                                   const std::vector<int64_t> &current_input_shape,
                                                   const std::vector<std::string> &input_names,
                                                   const std::unordered_map<std::string, std::vector<int64_t>> &shape_registry) const;
    std::vector<int64_t> infer_split_output_shape(const ExecutionStep &step,
                                                  const std::vector<int64_t> &current_input_shape,
                                                  size_t output_count) const;
    std::vector<int64_t> infer_split_output_shape_for_index(const ExecutionStep &step,
                                                            const std::vector<int64_t> &current_input_shape,
                                                            size_t output_count,
                                                            size_t output_index) const;
};
