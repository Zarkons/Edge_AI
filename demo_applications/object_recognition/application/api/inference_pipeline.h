#pragma once
#include "inference_pipeline_structures.h"
#include <string>
#include <vector>
#include <cstdint>
#include <string_view>
#include <map>
#include <deque>

namespace PipelineConstants
{
    // Academic Documentation: Clear string literal identifiers used to index the JSON map
    constexpr std::string_view OP_TYPE_KEY = "\"op_type\":";
    constexpr std::string_view SAFE_NAME_KEY = "\"cpp_safe_name\":";
    constexpr std::string_view WEIGHT_TENSORS_KEY = "\"weight_tensors\":";
    constexpr std::string_view EXECUTION_GRAPH_KEY = "\"execution_graph\":";
    constexpr std::string_view INPUTS_KEY = "\"inputs\":";
    constexpr std::string_view OUTPUTS_KEY = "\"outputs\":";
    constexpr std::string_view STRIDES_KEY = "\"strides\":";
    constexpr std::string_view PADS_KEY = "\"pads\":";
    constexpr std::string_view KERNEL_SHAPE_KEY = "\"kernel_shape\":";

    // Default spatial scaling metrics for YOLOv8n hardware channel tracks
    constexpr size_t STANDARD_SEQUENTIAL_STRIDE = 1;
    constexpr size_t DEFAULT_BACKBONE_CHANNEL_STRIDE = 64;
    constexpr int64_t DEFAULT_SPATIAL_HEIGHT = 640;
    constexpr int64_t DEFAULT_SPATIAL_WIDTH = 640;
}

class InferencePipeline
{
private:
    int weight_file_descriptor = -1;
    int execution_order_counter = 0;
    void *mmapped_weight_address = nullptr;
    size_t total_weight_buffer_bytes = 0;
    bool is_initialized = false;
    std::deque<std::vector<int64_t>> owned_constant_buffers;

public:
    InferencePipeline() = default;
    ~InferencePipeline();

    InferencePipeline(const InferencePipeline &) = delete;
    InferencePipeline &operator=(const InferencePipeline &) = delete;

    bool initialize_weight_buffer(const std::string &bin_path);
    void bind_tensor_view(WeightTensorView &view);

    bool load_and_optimize_manifest(const std::string &manifest_path,
                                    std::vector<WeightTensorView> &out_weights,
                                    std::vector<ExecutionStep> &out_pipeline);
    bool export_optimized_pipeline(const std::string &output_json_path,
                                   const std::vector<ExecutionStep> &pipeline);
    void verify_compiled_memory_bounds(const std::vector<ExecutionStep> &pipeline);
    bool ready() const { return is_initialized; }
    size_t get_buffer_size() const { return total_weight_buffer_bytes; }
    const void *get_base_address() const { return mmapped_weight_address; }

private:
    // PHASE 1: Manifest File and Weight Tensor Parsing
    bool load_manifest_file(const std::string &manifest_path, std::string &out_content);

    bool parse_weight_tensors(const std::string &json_content, size_t weight_start,
                              size_t weight_end, std::map<std::string, WeightTensorView> &registry,
                              std::vector<WeightTensorView> &out_weights);

    // PHASE 2: Execution Graph Parsing and Optimization
    bool parse_execution_graph(const std::string &json_content, size_t exec_start,
                               std::map<std::string, WeightTensorView> &weight_registry,
                               std::vector<ExecutionStep> &out_pipeline);

    // Helper methods for node processing
    void parse_node_io(const std::string &node_context, std::vector<std::string> &inputs,
                       std::vector<std::string> &outputs);

    std::string resolve_tensor_name(const std::string &name,
                                    const std::map<std::string, std::string> &rename_map);

    bool handle_quantization_ops(const std::string &op_type, const std::vector<std::string> &inputs,
                                 const std::vector<std::string> &outputs,
                                 std::map<std::string, std::string> &rename_map);

    void handle_split_slice_ops(const std::vector<std::string> &inputs,
                                const std::vector<std::string> &outputs,
                                const std::map<std::string, WeightTensorView> &weight_registry,
                                std::map<std::string, std::string> &rename_map,
                                size_t &out_accumulated_offset);

    void handle_reshape_transpose_ops(const std::string &op_type,
                                      const std::vector<std::string> &inputs,
                                      const std::vector<std::string> &outputs,
                                      const std::map<std::string, WeightTensorView> &weight_registry,
                                      std::map<std::string, std::string> &rename_map,
                                      size_t &out_stride_step);

    void handle_constant_op(const std::string &node_context,
                            const std::vector<std::string> &outputs,
                            std::map<std::string, WeightTensorView> &weight_registry);

    ExecutionStep create_execution_step(const std::string &op_type,
                                        const std::string &node_context,
                                        const std::vector<std::string> &inputs,
                                        const std::vector<std::string> &outputs,
                                        const std::map<std::string, WeightTensorView> &weight_registry,
                                        const std::map<std::string, std::string> &rename_map,
                                        const std::string &json_content, size_t accumulated_offset,
                                        size_t stride_step);
};
