#include "custom_inference_engine.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <limits>

CustomInferenceEngine::CustomInferenceEngine(const std::vector<int64_t> &root_input_shape,
                                             const std::string &root_input_name)
    : root_input_name_(root_input_name),
      root_input_shape_(root_input_shape)
{
    assert(!root_input_name_.empty() && "Root input name must not be empty.");
    assert(root_input_shape_.size() == 4 && "Root input shape must be [B, C, H, W].");
}

std::vector<int64_t> CustomInferenceEngine::infer_conv_output_shape(
    const ExecutionStep &step,
    const std::vector<int64_t> &current_input_shape) const
{
    int64_t batch = current_input_shape[0];
    int64_t in_h = current_input_shape[2];
    int64_t in_w = current_input_shape[3];

    int64_t out_channels = !step.kernel_shape.empty() ? step.kernel_shape[0] : 16;
    if (step.kernel_shape.size() >= 2)
    {
        out_channels = step.kernel_shape[0];
    }

    int64_t stride_h = step.strides.size() >= 1 ? step.strides[0] : 1;
    int64_t stride_w = step.strides.size() >= 2 ? step.strides[1] : 1;

    int64_t pad_top = step.pads.size() >= 1 ? step.pads[0] : 0;
    int64_t pad_left = step.pads.size() >= 2 ? step.pads[1] : 0;

    int64_t k_h = step.kernel_shape.size() >= 1 ? step.kernel_shape[step.kernel_shape.size() - 2] : 3;
    int64_t k_w = step.kernel_shape.size() >= 2 ? step.kernel_shape[step.kernel_shape.size() - 1] : 3;

    int64_t out_h = ((in_h + 2 * pad_top - k_h) / stride_h) + 1;
    int64_t out_w = ((in_w + 2 * pad_left - k_w) / stride_w) + 1;

    return {batch, out_channels, out_h, out_w};
}

std::vector<int64_t> CustomInferenceEngine::infer_maxpool_output_shape(
    const ExecutionStep &step,
    const std::vector<int64_t> &current_input_shape) const
{
    if (current_input_shape.size() < 4)
    {
        return current_input_shape;
    }

    int64_t batch = current_input_shape[0];
    int64_t channels = current_input_shape[1];
    int64_t in_h = current_input_shape[2];
    int64_t in_w = current_input_shape[3];

    int64_t stride_h = step.strides.size() >= 1 ? step.strides[0] : 1;
    int64_t stride_w = step.strides.size() >= 2 ? step.strides[1] : 1;

    int64_t pad_top = step.pads.size() >= 1 ? step.pads[0] : 0;
    int64_t pad_left = step.pads.size() >= 2 ? step.pads[1] : 0;

    int64_t k_h = step.kernel_shape.size() >= 1 ? step.kernel_shape[0] : 2;
    int64_t k_w = step.kernel_shape.size() >= 2 ? step.kernel_shape[1] : 2;

    double raw_h = static_cast<double>(in_h + 2 * pad_top - k_h) / static_cast<double>(stride_h);
    double raw_w = static_cast<double>(in_w + 2 * pad_left - k_w) / static_cast<double>(stride_w);

    int64_t out_h = (step.ceil_mode != 0) ? static_cast<int64_t>(std::ceil(raw_h) + 1.0) : static_cast<int64_t>(std::floor(raw_h) + 1.0);
    int64_t out_w = (step.ceil_mode != 0) ? static_cast<int64_t>(std::ceil(raw_w) + 1.0) : static_cast<int64_t>(std::floor(raw_w) + 1.0);

    out_h = std::max<int64_t>(1, out_h);
    out_w = std::max<int64_t>(1, out_w);

    return {batch, channels, out_h, out_w};
}

std::vector<int64_t> CustomInferenceEngine::infer_concat_output_shape(
    const ExecutionStep &step,
    const std::vector<int64_t> &current_input_shape,
    const std::vector<std::string> &input_names,
    const std::unordered_map<std::string, std::vector<int64_t>> &shape_registry) const
{
    std::vector<int64_t> concat_shape = current_input_shape;
    if (concat_shape.empty())
    {
        return concat_shape;
    }

    const int64_t rank = static_cast<int64_t>(concat_shape.size());
    const int64_t axis = (step.axis < 0) ? (step.axis + rank) : step.axis;
    if (axis < 0 || axis >= rank)
    {
        return concat_shape;
    }

    int64_t axis_sum = 0;
    for (const auto &in_name : input_names)
    {
        auto it = shape_registry.find(in_name);
        if (it != shape_registry.end() && it->second.size() == concat_shape.size())
        {
            axis_sum += it->second[axis];
        }
    }

    if (axis_sum > 0)
    {
        concat_shape[axis] = axis_sum;
    }

    return concat_shape;
}

std::vector<int64_t> CustomInferenceEngine::infer_split_output_shape(
    const ExecutionStep &step,
    const std::vector<int64_t> &current_input_shape,
    size_t output_count) const
{
    std::vector<int64_t> split_shape = current_input_shape;
    if (split_shape.empty() || output_count == 0)
    {
        return split_shape;
    }

    const int64_t rank = static_cast<int64_t>(split_shape.size());
    const int64_t axis = (step.axis < 0) ? (step.axis + rank) : step.axis;
    if (axis < 0 || axis >= rank)
    {
        return split_shape;
    }

    if (split_shape[axis] >= static_cast<int64_t>(output_count))
    {
        split_shape[axis] = split_shape[axis] / static_cast<int64_t>(output_count);
    }

    return split_shape;
}

std::vector<int64_t> CustomInferenceEngine::infer_split_output_shape_for_index(
    const ExecutionStep &step,
    const std::vector<int64_t> &current_input_shape,
    size_t output_count,
    size_t output_index) const
{
    std::vector<int64_t> split_shape = infer_split_output_shape(step, current_input_shape, output_count);
    if (split_shape.empty() || output_count == 0)
    {
        return split_shape;
    }

    const int64_t rank = static_cast<int64_t>(split_shape.size());
    const int64_t axis = (step.axis < 0) ? (step.axis + rank) : step.axis;
    if (axis < 0 || axis >= rank)
    {
        return split_shape;
    }

    const int64_t source_dim = current_input_shape[axis];
    const int64_t outputs_count_i64 = static_cast<int64_t>(output_count);
    const int64_t base = source_dim / outputs_count_i64;
    const int64_t remainder = source_dim % outputs_count_i64;

    split_shape[axis] = base + ((static_cast<int64_t>(output_index) < remainder) ? 1 : 0);
    return split_shape;
}

std::vector<int64_t> CustomInferenceEngine::infer_output_shape_for_step(
    const ExecutionStep &step,
    const std::vector<std::vector<int64_t>> &resolved_input_shapes, // Updated parameter type
    const std::vector<std::string> &input_names,
    const std::unordered_map<std::string, std::vector<int64_t>> &shape_registry) const
{
    // Establish a safe primary fallback shape from the first available resolved shape
    std::vector<int64_t> primary_shape = root_input_shape_;
    for (const auto &shape : resolved_input_shapes)
    {
        if (!shape.empty())
        {
            primary_shape = shape;
            break;
        }
    }

    if (step.op_type == "Conv")
    {
        return infer_conv_output_shape(step, primary_shape);
    }
    else if (step.op_type == "MaxPool")
    {
        return infer_maxpool_output_shape(step, primary_shape);
    }
    else if (step.op_type == "Concat")
    {
        // Concat now directly uses the resolved list or its internal registry logic
        return infer_concat_output_shape(step, primary_shape, input_names, shape_registry);
    }
    else if (step.op_type == "Sigmoid" || step.op_type == "Resize")
    {
        return primary_shape; // Unary ops pass geometry straight through
    }
    else if (step.op_type == "Mul" || step.op_type == "Add" || step.op_type == "Sub" || step.op_type == "Div")
    {
        // For element-wise operations, the highest rank tensor dictates the broadcast output shape
        std::vector<int64_t> broadcast_shape = primary_shape;
        for (const auto &shape : resolved_input_shapes)
        {
            if (shape.size() > broadcast_shape.size())
            {
                broadcast_shape = shape;
            }
        }
        return broadcast_shape;
    }

    return primary_shape;
}

// --- Ahead-of-Time Runtime Plan Initialization and Shape Inference Pass ---

size_t CustomInferenceEngine::initialize_runtime_plan(std::vector<ExecutionStep> &blueprints)
{
    // 1. Reset your offset registries and pre-allocate the plan vector size
    arena_offset_map_.clear();
    execution_plan_.clear();
    execution_plan_.reserve(blueprints.size());

    // 2. Track dynamic activation shapes across interconnected layers using constructor-seeded defaults
    std::unordered_map<std::string, std::vector<int64_t>> shape_registry;
    shape_registry[root_input_name_] = root_input_shape_;

    size_t relative_offset_tracker = 0;

    // Cache the original step string dependencies before moving/draining them
    std::vector<std::vector<std::string>> original_inputs_map;
    std::vector<std::vector<std::string>> original_outputs_map;
    original_inputs_map.reserve(blueprints.size());
    original_outputs_map.reserve(blueprints.size());

    for (const auto &step : blueprints)
    {
        original_inputs_map.push_back(step.inputs);
        original_outputs_map.push_back(step.outputs);
    }

    // Single unified loop over our incoming execution sequence arrays
    for (size_t i = 0; i < blueprints.size(); ++i)
    {
        auto &step = blueprints[i];

        std::vector<std::vector<int64_t>> resolved_input_shapes;
        resolved_input_shapes.reserve(original_inputs_map[i].size());

        size_t primary_lane_index = 0;
        bool found_primary = false;

        for (size_t lane_idx = 0; lane_idx < original_inputs_map[i].size(); ++lane_idx)
        {
            const auto &in_name = original_inputs_map[i][lane_idx];
            auto it = shape_registry.find(in_name);
            if (it != shape_registry.end())
            {
                resolved_input_shapes.push_back(it->second);

                // SKIPPING CONSTANTS: Prioritize tracking non-weight/non-bias tensors
                bool is_constant = (in_name.find("weight") != std::string::npos ||
                                    in_name.find("bias") != std::string::npos);

                if (!is_constant && !found_primary)
                {
                    primary_lane_index = lane_idx;
                    found_primary = true;
                }
                // FALLBACK: If dimensions match or look unusual, prefer the tensor with the highest rank
                else if (!found_primary && (resolved_input_shapes.back().size() > resolved_input_shapes[primary_lane_index].size()))
                {
                    primary_lane_index = lane_idx;
                }
            }
            else
            {
                resolved_input_shapes.push_back({});
            }
        }

        // Handle a scenario where none of the inputs matched known registry profiles
        if (resolved_input_shapes.empty() || resolved_input_shapes[primary_lane_index].empty())
        {
            // If empty, force a fresh placeholder slot matching our root tracking context
            resolved_input_shapes.push_back(root_input_shape_);
            primary_lane_index = resolved_input_shapes.size() - 1;
        }

        // Step B: Infer the Output Shape based on operational type behavior rules
        std::vector<int64_t> inferred_output_shape = infer_output_shape_for_step(
            step,
            resolved_input_shapes,
            original_inputs_map[i],
            shape_registry);

        // Step C: Map Output Names to Registry and calculate safe Memory Arena Offsets
        for (size_t out_idx = 0; out_idx < original_outputs_map[i].size(); ++out_idx)
        {
            const std::string &out_name = original_outputs_map[i][out_idx];
            std::vector<int64_t> per_output_shape = inferred_output_shape;

            shape_registry[out_name] = per_output_shape;

            if (arena_offset_map_.find(out_name) == arena_offset_map_.end())
            {
                if (relative_offset_tracker % 16 != 0)
                {
                    relative_offset_tracker += (16 - (relative_offset_tracker % 16));
                }

                arena_offset_map_[out_name] = relative_offset_tracker;

                size_t dynamic_elements = 1;
                for (int64_t dimension : per_output_shape)
                {
                    dynamic_elements *= dimension;
                }

                // To be changed to int8 after DequantizeLinear is removed from the graph. For now, we assume float32 output for all ops.
                size_t width_bytes = sizeof(float);
                relative_offset_tracker += (dynamic_elements * width_bytes);
            }
        }

        // Step D: Instantiate a pre-linked Runtime Node mapping snapshot onto our queue plan
        RuntimeNode node;
        node.op_type = step.op_type;
        node.input_shapes = std::move(resolved_input_shapes);
        node.primary_shape_idx = primary_lane_index;
        node.output_shape = inferred_output_shape;

        node.input_arena_offsets.reserve(original_inputs_map[i].size());
        node.output_arena_offsets.reserve(original_outputs_map[i].size());

        for (const auto &in_name : original_inputs_map[i])
        {
            if (in_name.find("weight") != std::string::npos || in_name.find("bias") != std::string::npos || in_name == root_input_name_)
            {
                node.input_arena_offsets.push_back(std::numeric_limits<size_t>::max());
            }
            else
            {
                node.input_arena_offsets.push_back(arena_offset_map_[in_name]);
            }
        }

        for (const auto &out_name : original_outputs_map[i])
        {
            node.output_arena_offsets.push_back(arena_offset_map_[out_name]);
        }

        // ZERO-COPY CONVERSION: Directly move data structures, emptying user allocation loads safely
        node.blueprint = std::move(step);
        execution_plan_.push_back(std::move(node));
    }

    // Clear the user's vector to explicitly finalize our zero-copy memory transfer handoff
    blueprints.clear();

    arena_size_ = relative_offset_tracker;
    return arena_size_;
}

// --- Live Execution Pass (Zero-Allocation Forward Loop) ---

void CustomInferenceEngine::forward(const void *input_hardware_buffer,
                                    void *output_hardware_buffer,
                                    void *arena_workspace)
{
    assert(!execution_plan_.empty() && "Inference pipeline must be initialized prior to invoking forward loops.");
    assert(arena_workspace && "Valid pre-allocated user arena workspace buffer address must be provided.");

    uint8_t *arena_base = reinterpret_cast<uint8_t *>(arena_workspace);

    for (size_t i = 0; i < execution_plan_.size(); ++i)
    {
        const auto &node = execution_plan_[i];

        // 1. Set up input address lanes on the stack based on operation type indices
        std::vector<const void *> input_lanes;
        input_lanes.reserve(node.input_arena_offsets.size());

        for (size_t lane_idx = 0; lane_idx < node.input_arena_offsets.size(); ++lane_idx)
        {
            const size_t offset = node.input_arena_offsets[lane_idx];
            if (offset != std::numeric_limits<size_t>::max())
            {
                input_lanes.push_back(reinterpret_cast<const void *>(arena_base + offset));
            }
            else if (lane_idx < node.blueprint.inputs.size() && node.blueprint.inputs[lane_idx] == root_input_name_)
            {
                input_lanes.push_back(input_hardware_buffer);
            }
            else
            {
                input_lanes.push_back(nullptr);
            }
        }

        // 2. Set up output address lanes on the stack
        void *act_out = nullptr;
        if (i == execution_plan_.size() - 1)
        {
            act_out = output_hardware_buffer;
        }
        else
        {
            size_t offset = node.output_arena_offsets[0];
            act_out = reinterpret_cast<void *>(arena_base + offset);
        }

        // 3. Delegate op routing and execution to CustomKernels.
        kernels_.execute(node,
                         input_lanes,
                         act_out);
    }
}
