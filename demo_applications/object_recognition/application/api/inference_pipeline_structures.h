#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct WeightTensorView
{
    std::string name;
    std::string cpp_safe_name;
    std::vector<int64_t> shape;
    std::string data_type;
    size_t byte_offset = 0;
    size_t byte_length = 0;

    const int8_t *int8_ptr = nullptr;
    const float *fp32_ptr = nullptr;
};

// Expanded Execution Step containing fused quantization and structural trackers
struct ExecutionStep
{
    int execution_order = 0;
    std::string op_type;
    std::string node_name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;

    // Core layer attributes from graph
    int64_t axis = 0;
    int64_t ceil_mode = 0;
    float cubic_coeff_a = -0.75f;
    std::vector<int64_t> dilations;
    int64_t group = 1;
    std::vector<int64_t> strides;
    std::vector<int64_t> pads;
    std::vector<int64_t> kernel_shape;

    // FUSED OPERATOR WEIGHT POINTERS (Mapped directly via mmap)
    const int8_t *weights_ptr = nullptr;
    const int32_t *bias_ptr = nullptr;

    // FUSED QUANTIZATION PARAMETERS (Extracted during manifest parsing)
    float input_scale = 1.0f;
    float weight_scale = 1.0f;
    float output_scale = 1.0f;
    int32_t input_zero_point = 0;
    int32_t weight_zero_point = 0;
    int32_t output_zero_point = 0;

    // FUSED STRIDE, LAYOUT AND OFFSET VIEW TRACKERS
    size_t input_byte_offset = 0;
    size_t custom_stride_step = 1;
};
