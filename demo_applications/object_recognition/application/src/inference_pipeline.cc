#include "inference_pipeline.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <iomanip>
#include <cstring>
#include <algorithm>

// Low-level helper to manually extract integer arrays from JSON blocks (e.g. shapes)
std::vector<int64_t> parse_json_int_array(const std::string &source, const std::string &key)
{
    std::vector<int64_t> result;

    // 1. Find just the clean quoted key wrapper (e.g., "strides")
    size_t key_pos = source.find("\"" + key + "\"");
    if (key_pos == std::string::npos)
        return result;

    // 2. Scan forward past the key to find the opening bracket safely
    // This allows arbitrary spaces, colons, or even newlines between the key and its values
    size_t start_bracket = std::string::npos;
    for (size_t i = key_pos + key.size() + 2; i < source.size(); ++i)
    {
        if (source[i] == '[')
        {
            start_bracket = i;
            break;
        }
        if (source[i] == '}')
            break; // Scope safety boundary escape
    }

    size_t end_bracket = source.find(']', start_bracket);
    if (start_bracket == std::string::npos || end_bracket == std::string::npos || start_bracket >= end_bracket)
        return result;

    // 3. Process the internal array tokens securely
    std::string array_content = source.substr(start_bracket + 1, end_bracket - start_bracket - 1);
    std::stringstream ss(array_content);
    std::string token;

    while (std::getline(ss, token, ','))
    {
        // Erase ALL formatting layout clutter, tabs, spaces, or lines from the numeric element
        token.erase(std::remove_if(token.begin(), token.end(), [](char c)
                                   { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }),
                    token.end());

        if (token.empty())
            continue;

        try
        {
            // Now safe to parse because strings are completely clean
            result.push_back(std::stoll(token));
        }
        catch (...)
        {
            // Safeguard tracking for debugging unexpected character noise
        }
    }
    return result;
}

// Low-level helper to extract clean string value fields out of JSON keys
std::string parse_json_string_value(const std::string &source, const std::string &key, size_t search_start)
{
    size_t key_pos = source.find("\"" + key + "\":", search_start);
    if (key_pos == std::string::npos)
        return "";

    size_t start_quote = source.find('\"', key_pos + key.length() + 3);
    size_t end_quote = source.find('\"', start_quote + 1);
    if (start_quote == std::string::npos || end_quote == std::string::npos)
        return "";

    return source.substr(start_quote + 1, end_quote - start_quote - 1);
}

// Robust substring search helper to capture numeric scalar values safely
int64_t parse_json_scalar_int(const std::string &source, const std::string &key, size_t search_start,
                              int64_t default_value = 0)
{
    size_t key_pos = source.find("\"" + key + "\":", search_start);
    if (key_pos == std::string::npos)
        return default_value;

    size_t colon_pos = source.find(':', key_pos);
    if (colon_pos == std::string::npos)
        return default_value;

    size_t value_start = source.find_first_not_of(" \t\n\r", colon_pos + 1);
    if (value_start == std::string::npos)
        return default_value;

    size_t scan_idx = value_start;
    if (source[scan_idx] == '-')
        ++scan_idx;

    size_t digit_start = scan_idx;
    while (scan_idx < source.size() && std::isdigit(static_cast<unsigned char>(source[scan_idx])))
        ++scan_idx;

    if (digit_start == scan_idx)
        return default_value;

    try
    {
        return std::stoll(source.substr(value_start, scan_idx - value_start));
    }
    catch (...)
    {
        return default_value;
    }
}

InferencePipeline::~InferencePipeline()
{
    if (mmapped_weight_address != MAP_FAILED && mmapped_weight_address != nullptr)
    {
        munmap(mmapped_weight_address, total_weight_buffer_bytes);
        std::cout << "🟩 Successfully unmapped weight flat buffer from memory space.\n";
    }
    if (weight_file_descriptor != -1)
    {
        close(weight_file_descriptor);
    }
}

bool InferencePipeline::initialize_weight_buffer(const std::string &bin_path)
{
    weight_file_descriptor = open(bin_path.c_str(), O_RDONLY);
    if (weight_file_descriptor == -1)
    {
        std::cerr << "🟥 Failed to open binary weight payload file: " << bin_path << "\n";
        return false;
    }

    struct stat file_statistics;
    if (fstat(weight_file_descriptor, &file_statistics) == -1)
    {
        std::cerr << "🟥 Failed to read file allocation stats.\n";
        close(weight_file_descriptor);
        weight_file_descriptor = -1;
        return false;
    }
    total_weight_buffer_bytes = file_statistics.st_size;

    mmapped_weight_address = mmap(nullptr, total_weight_buffer_bytes, PROT_READ, MAP_SHARED, weight_file_descriptor, 0);
    if (mmapped_weight_address == MAP_FAILED)
    {
        std::cerr << "🟥 Low-level memory map execution failed!\n";
        close(weight_file_descriptor);
        weight_file_descriptor = -1;
        return false;
    }

    is_initialized = true;
    std::cout << "🟩 ZERO-COPY SUCCESS: Memory-mapped "
              << (total_weight_buffer_bytes / (1024.0 * 1024.0))
              << " MB of weights directly to address space point: "
              << mmapped_weight_address << "\n";
    return true;
}

void InferencePipeline::bind_tensor_view(WeightTensorView &view)
{
    if (!is_initialized)
        return;

    uintptr_t base_address = reinterpret_cast<uintptr_t>(mmapped_weight_address);
    uintptr_t target_address = base_address + view.byte_offset;

    if (view.data_type == "float" || view.data_type == "float32")
    {
        view.fp32_ptr = reinterpret_cast<const float *>(target_address);
    }
    else
    {
        // Keep a raw byte pointer for all non-float tensors (int8/int32/uint8, etc.).
        view.int8_ptr = reinterpret_cast<const int8_t *>(target_address);
    }
}

// Helper to locate scalar scale parameters directly out of the JSON string buffer
float parse_json_scalar_float(const std::string &source, const std::string &key,
                              float default_value = 1.0f)
{
    size_t key_pos = source.find("\"" + key + "\":");
    if (key_pos == std::string::npos)
        return default_value;

    size_t colon_pos = source.find(':', key_pos);
    if (colon_pos == std::string::npos)
        return default_value;

    size_t value_start = source.find_first_not_of(" \t\n\r", colon_pos + 1);
    if (value_start == std::string::npos)
        return default_value;

    size_t value_end = value_start;
    while (value_end < source.size())
    {
        const char c = source[value_end];
        const bool allowed = std::isdigit(static_cast<unsigned char>(c)) ||
                             c == '-' || c == '+' || c == '.' ||
                             c == 'e' || c == 'E';
        if (!allowed)
            break;
        ++value_end;
    }

    if (value_end == value_start)
        return default_value;

    try
    {
        return std::stof(source.substr(value_start, value_end - value_start));
    }
    catch (...)
    {
        return default_value;
    }
}

bool InferencePipeline::load_and_optimize_manifest(const std::string &manifest_path,
                                                   std::vector<WeightTensorView> &out_weights,
                                                   std::vector<ExecutionStep> &out_pipeline)
{
    // Load manifest JSON file
    std::string json_content;
    if (!load_manifest_file(manifest_path, json_content))
        return false;

    // Locate section boundaries in manifest
    size_t weight_section_pos = json_content.find(PipelineConstants::WEIGHT_TENSORS_KEY);
    size_t execution_section_pos = json_content.find(PipelineConstants::EXECUTION_GRAPH_KEY);

    if (weight_section_pos == std::string::npos || execution_section_pos == std::string::npos)
    {
        std::cerr << "Malforming manifest context boundaries.\n";
        return false;
    }

    // Parse weight tensors (PHASE 1)
    std::map<std::string, WeightTensorView> weight_registry;
    if (!parse_weight_tensors(json_content, weight_section_pos, execution_section_pos,
                              weight_registry, out_weights))
        return false;

    // Parse execution graph (PHASE 2)
    if (!parse_execution_graph(json_content, execution_section_pos, weight_registry, out_pipeline))
        return false;

    std::cout << "🟩 BARE-METAL GRAPH STITCHING COMPLETE: 212 Math Operations Baked.\n";
    return true;
}

// ============================================================================
// PHASE 1: Manifest File and Weight Tensor Parsing
// ============================================================================

bool InferencePipeline::load_manifest_file(const std::string &manifest_path, std::string &out_content)
{
    std::ifstream file(manifest_path);
    if (!file.is_open())
    {
        std::cerr << "File access missing: " << manifest_path << "\n";
        return false;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    out_content = buffer.str();
    return true;
}

bool InferencePipeline::parse_weight_tensors(const std::string &json_content, size_t weight_start,
                                             size_t weight_end,
                                             std::map<std::string, WeightTensorView> &registry,
                                             std::vector<WeightTensorView> &out_weights)
{
    size_t section_open = json_content.find('{', weight_start);
    if (section_open == std::string::npos || section_open >= weight_end)
        return false;

    size_t current_search_idx = section_open + 1;

    while (current_search_idx < weight_end)
    {
        size_t key_start = json_content.find('"', current_search_idx);
        if (key_start == std::string::npos || key_start >= weight_end)
            break;

        size_t key_end = json_content.find('"', key_start + 1);
        if (key_end == std::string::npos || key_end >= weight_end)
            break;

        std::string raw_tensor_name = json_content.substr(key_start + 1, key_end - key_start - 1);

        size_t colon_pos = json_content.find(':', key_end + 1);
        if (colon_pos == std::string::npos || colon_pos >= weight_end)
            break;

        size_t block_start = json_content.find('{', colon_pos + 1);
        if (block_start == std::string::npos || block_start >= weight_end)
        {
            current_search_idx = key_end + 1;
            continue;
        }

        size_t block_end = json_content.find('}', block_start + 1);
        if (block_end == std::string::npos || block_end >= weight_end)
            break;

        std::string context_block = json_content.substr(block_start, block_end - block_start + 1);

        // Parse tensor metadata
        WeightTensorView view;
        view.name = raw_tensor_name;
        view.cpp_safe_name = parse_json_string_value(context_block, "cpp_safe_name", 0);
        view.data_type = parse_json_string_value(context_block, "data_type", 0);

        view.byte_offset = parse_json_scalar_int(context_block, "byte_offset", 0);
        view.byte_length = parse_json_scalar_int(context_block, "byte_length", 0);
        view.shape = parse_json_int_array(context_block, "shape");

        // Bind memory pointer and register the tensor
        this->bind_tensor_view(view);
        registry[raw_tensor_name] = view;
        if (!view.cpp_safe_name.empty())
        {
            registry[view.cpp_safe_name] = view;
        }
        out_weights.push_back(view);

        current_search_idx = block_end + 1;
    }

    return true;
}

// ============================================================================
// PHASE 2: Execution Graph Parsing and Optimization
// ============================================================================

bool InferencePipeline::parse_execution_graph(const std::string &json_content, size_t exec_start,
                                              std::map<std::string, WeightTensorView> &weight_registry,
                                              std::vector<ExecutionStep> &out_pipeline)
{
    std::map<std::string, std::string> rename_map; // Maps layer names through quantization/reshape ops
    size_t current_search_idx = exec_start;
    size_t accumulated_offset = 0;
    size_t stride_step = PipelineConstants::STANDARD_SEQUENTIAL_STRIDE;

    while (true)
    {
        size_t op_pos = json_content.find(PipelineConstants::OP_TYPE_KEY, current_search_idx);
        if (op_pos == std::string::npos)
            break;

        std::string op_type = parse_json_string_value(json_content, "op_type", op_pos);
        size_t node_end = json_content.find("}", op_pos);
        std::string node_context = json_content.substr(op_pos, node_end - op_pos);

        // Parse node inputs and outputs
        std::vector<std::string> inputs, outputs;
        parse_node_io(node_context, inputs, outputs);

        // Process node based on operation type
        if (handle_quantization_ops(op_type, inputs, outputs, rename_map))
        {
            // Quantization ops are skipped but mapped
            current_search_idx = node_end;
            continue;
        }

        if (op_type == "Split" || op_type == "Slice")
        {
            handle_split_slice_ops(inputs, outputs, weight_registry, rename_map, accumulated_offset);
            current_search_idx = node_end;
            continue;
        }

        if (op_type == "Reshape" || op_type == "Transpose")
        {
            handle_reshape_transpose_ops(op_type, inputs, outputs, weight_registry, rename_map, stride_step);
            current_search_idx = node_end;
            continue;
        }

        if (op_type == "Constant")
        {
            handle_constant_op(node_context, outputs, weight_registry);
            current_search_idx = node_end;
            continue;
        }

        // Create execution step for computational nodes
        ExecutionStep step = create_execution_step(op_type, node_context, inputs, outputs,
                                                   weight_registry, rename_map, json_content,
                                                   accumulated_offset, stride_step);

        // Reset state registers for next iteration
        accumulated_offset = 0;
        stride_step = PipelineConstants::STANDARD_SEQUENTIAL_STRIDE;

        out_pipeline.push_back(step);
        current_search_idx = node_end;
    }

    return true;
}

// ============================================================================
// Helper Methods for Node Processing
// ============================================================================

void InferencePipeline::parse_node_io(const std::string &node_context,
                                      std::vector<std::string> &inputs,
                                      std::vector<std::string> &outputs)
{
    // A consolidated lambda utility ensures unified cleaning behavior across both arrays
    auto extract_and_clean_array = [](const std::string &context, std::string_view key, std::vector<std::string> &target_vec)
    {
        size_t key_pos = context.find(key);
        if (key_pos == std::string::npos)
            return;

        size_t start_brac = context.find('[', key_pos);
        size_t end_brac = context.find(']', key_pos);
        if (start_brac == std::string::npos || end_brac == std::string::npos || start_brac >= end_brac)
            return;

        std::stringstream ss(context.substr(start_brac + 1, end_brac - start_brac - 1));
        std::string token;

        while (std::getline(ss, token, ','))
        {
            // Erase ALL possible white-spaces, formatting escapes, layout tabs, and quotation wrappers
            token.erase(std::remove_if(token.begin(), token.end(), [](char c)
                                       { return c == '\"' || c == ' ' || c == '\n' ||
                                                c == '\r' || c == '\t' || c == '\f' || c == '\v'; }),
                        token.end());

            if (!token.empty())
            {
                target_vec.push_back(token);
            }
        }
    };

    // Cleanly populate your IO registers with sanitized string assets
    extract_and_clean_array(node_context, PipelineConstants::INPUTS_KEY, inputs);
    extract_and_clean_array(node_context, PipelineConstants::OUTPUTS_KEY, outputs);
}

std::string InferencePipeline::resolve_tensor_name(const std::string &name,
                                                   const std::map<std::string, std::string> &rename_map)
{
    // Follow the rename map chain to find the true source tensor name
    std::string current = name;
    while (rename_map.find(current) != rename_map.end())
    {
        current = rename_map.at(current);
    }
    return current;
}

bool InferencePipeline::handle_quantization_ops(const std::string &op_type,
                                                const std::vector<std::string> &inputs,
                                                const std::vector<std::string> &outputs,
                                                std::map<std::string, std::string> &rename_map)
{
    // Quantization/Dequantization ops are format wrappers that map input to output
    // but don't perform computation themselves
    if (op_type != "DequantizeLinear" && op_type != "QuantizeLinear")
        return false;

    if (!inputs.empty() && !outputs.empty())
    {
        // Map output tensor name to input tensor name
        for (const auto &output : outputs)
        {
            rename_map[output] = inputs[0];
        }
    }
    return true;
}

void InferencePipeline::handle_split_slice_ops(const std::vector<std::string> &inputs,
                                               const std::vector<std::string> &outputs,
                                               const std::map<std::string, WeightTensorView> &weight_registry,
                                               std::map<std::string, std::string> &rename_map,
                                               size_t &out_accumulated_offset)
{
    if (!inputs.empty())
    {
        // Trace back to the true source tensor through the rename map
        std::string baseline_input = resolve_tensor_name(inputs[0], rename_map);

        int64_t spatial_height = PipelineConstants::DEFAULT_SPATIAL_HEIGHT;
        int64_t spatial_width = PipelineConstants::DEFAULT_SPATIAL_WIDTH;

        // Query weight registry for dynamic spatial dimensions
        auto it = weight_registry.find(baseline_input);
        if (it != weight_registry.end())
        {
            const auto &parent_shape = it->second.shape;
            // For 4D tensors [B, C, H, W], extract height and width
            if (parent_shape.size() >= 2)
            {
                spatial_height = parent_shape[parent_shape.size() - 2];
                spatial_width = parent_shape[parent_shape.size() - 1];
            }
        }

        // Accumulate the layout offset based on spatial dimensions
        out_accumulated_offset += static_cast<size_t>(spatial_height * spatial_width);
    }

    // Register output mappings
    if (!inputs.empty() && !outputs.empty())
    {
        for (const auto &output : outputs)
        {
            rename_map[output] = inputs[0];
        }
    }
}

void InferencePipeline::handle_reshape_transpose_ops(const std::string &op_type,
                                                     const std::vector<std::string> &inputs,
                                                     const std::vector<std::string> &outputs,
                                                     const std::map<std::string, WeightTensorView> &weight_registry,
                                                     std::map<std::string, std::string> &rename_map,
                                                     size_t &out_stride_step)
{
    if (op_type == "Transpose" && !inputs.empty())
    {
        // For Transpose, update stride based on source tensor channel dimension
        std::string baseline_input = resolve_tensor_name(inputs[0], rename_map);

        auto it = weight_registry.find(baseline_input);
        if (it != weight_registry.end())
        {
            const auto &parent_shape = it->second.shape;
            if (parent_shape.size() >= 2)
            {
                // Channel dimension is typically at index 1
                out_stride_step = static_cast<size_t>(parent_shape[1]);
            }
        }
        else
        {
            out_stride_step = PipelineConstants::DEFAULT_BACKBONE_CHANNEL_STRIDE;
        }
    }

    // Register output mappings for both Reshape and Transpose
    if (!inputs.empty() && !outputs.empty())
    {
        for (const auto &output : outputs)
        {
            rename_map[output] = inputs[0];
        }
    }
}

void InferencePipeline::handle_constant_op(const std::string &node_context,
                                           const std::vector<std::string> &outputs,
                                           std::map<std::string, WeightTensorView> &weight_registry)
{
    if (outputs.empty())
        return;

    const std::string constant_tensor_name = outputs[0];
    std::vector<int64_t> constant_values = parse_json_int_array(node_context, "value");
    if (constant_values.empty())
        return;

    // Keep parsed constants alive for the lifetime of the engine.
    owned_constant_buffers.push_back(std::move(constant_values));
    const auto &constant_buffer = owned_constant_buffers.back();

    WeightTensorView constant_view;
    constant_view.name = constant_tensor_name;
    constant_view.shape = {static_cast<int64_t>(constant_buffer.size())};
    constant_view.data_type = "int64";
    constant_view.byte_length = constant_buffer.size() * sizeof(int64_t);
    constant_view.int8_ptr = reinterpret_cast<const int8_t *>(constant_buffer.data());

    weight_registry[constant_tensor_name] = constant_view;
}

ExecutionStep InferencePipeline::create_execution_step(const std::string &op_type,
                                                       const std::string &node_context,
                                                       const std::vector<std::string> &inputs,
                                                       const std::vector<std::string> &outputs,
                                                       const std::map<std::string, WeightTensorView> &weight_registry,
                                                       const std::map<std::string, std::string> &rename_map,
                                                       const std::string &json_content,
                                                       size_t accumulated_offset,
                                                       size_t stride_step)
{
    ExecutionStep step;
    step.execution_order = execution_order_counter++;
    step.op_type = op_type;
    step.node_name = parse_json_string_value(node_context, "node_name", 0);

    auto read_scale_from_tensor = [](const WeightTensorView &scale_view) -> float
    {
        if (scale_view.fp32_ptr)
            return *scale_view.fp32_ptr;
        if (scale_view.int8_ptr && scale_view.byte_length >= sizeof(float))
        {
            float value = 1.0f;
            std::memcpy(&value, scale_view.int8_ptr, sizeof(float));
            return value;
        }
        return 1.0f;
    };

    auto read_zero_point_from_tensor = [](const WeightTensorView &zp_view) -> int32_t
    {
        if (!zp_view.int8_ptr)
            return 0;

        if (zp_view.data_type == "uint8" || zp_view.data_type == "uint8_t")
            return static_cast<int32_t>(*reinterpret_cast<const uint8_t *>(zp_view.int8_ptr));

        if (zp_view.data_type == "int8" || zp_view.data_type == "int8_t")
            return static_cast<int32_t>(*reinterpret_cast<const int8_t *>(zp_view.int8_ptr));

        if (zp_view.data_type == "int32" || zp_view.data_type == "int32_t" ||
            zp_view.byte_length >= sizeof(int32_t))
        {
            int32_t value = 0;
            std::memcpy(&value, zp_view.int8_ptr, sizeof(int32_t));
            return value;
        }

        if (zp_view.byte_length == 1)
            return static_cast<int32_t>(*reinterpret_cast<const int8_t *>(zp_view.int8_ptr));

        return 0;
    };

    auto with_quantized_suffix_removed = [](const std::string &name) -> std::string
    {
        constexpr size_t quantized_suffix_len = 10;
        if (name.size() > quantized_suffix_len &&
            name.substr(name.size() - quantized_suffix_len) == "_quantized")
        {
            return name.substr(0, name.size() - quantized_suffix_len);
        }
        return name;
    };

    auto find_quant_tensor = [&](const std::string &tensor_name,
                                 const std::string &suffix) -> std::map<std::string, WeightTensorView>::const_iterator
    {
        auto it = weight_registry.find(tensor_name + suffix);
        if (it != weight_registry.end())
            return it;

        std::string base_name = with_quantized_suffix_removed(tensor_name);
        it = weight_registry.find(base_name + suffix);
        if (it != weight_registry.end())
            return it;

        return weight_registry.end();
    };

    auto try_find_tensor = [&](const std::string &name) -> std::map<std::string, WeightTensorView>::const_iterator
    {
        auto it = weight_registry.find(name);
        if (it != weight_registry.end())
            return it;

        constexpr size_t quantized_suffix_len = 10;
        if (name.size() > quantized_suffix_len &&
            name.substr(name.size() - quantized_suffix_len) == "_quantized")
        {
            std::string base_name = name.substr(0, name.size() - quantized_suffix_len);
            return weight_registry.find(base_name);
        }

        return weight_registry.end();
    };

    // Track input index to route scale properties to their exact functional parameters
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        std::string resolved_input = resolve_tensor_name(inputs[i], rename_map);
        step.inputs.push_back(resolved_input);

        // Gather quantization scale/zero-point for activation and weight inputs.
        if (i <= 1)
        {
            auto scale_it = find_quant_tensor(resolved_input, "_scale");
            auto zp_it = find_quant_tensor(resolved_input, "_zero_point");

            float parsed_scale = 1.0f;
            int32_t parsed_zero_point = 0;

            if (scale_it != weight_registry.end())
                parsed_scale = read_scale_from_tensor(scale_it->second);
            if (zp_it != weight_registry.end())
                parsed_zero_point = read_zero_point_from_tensor(zp_it->second);

            if (i == 0)
            {
                step.input_scale = parsed_scale;
                step.input_zero_point = parsed_zero_point;
            }
            else
            {
                step.weight_scale = parsed_scale;
                step.weight_zero_point = parsed_zero_point;
            }
        }
    }

    // Add output tensors
    for (const auto &output : outputs)
    {
        step.outputs.push_back(output);
    }

    if (!step.outputs.empty())
    {
        auto output_scale_it = find_quant_tensor(step.outputs[0], "_scale");
        auto output_zp_it = find_quant_tensor(step.outputs[0], "_zero_point");

        if (output_scale_it != weight_registry.end())
            step.output_scale = read_scale_from_tensor(output_scale_it->second);

        if (output_zp_it != weight_registry.end())
            step.output_zero_point = read_zero_point_from_tensor(output_zp_it->second);
    }

    // --- WEIGHT & BIAS POINTER BINDING ---
    // Handle the primary weight matrix pointer mapping
    if (step.inputs.size() >= 2)
    {
        auto weight_it = try_find_tensor(step.inputs[1]);
        if (weight_it != weight_registry.end())
        {
            step.weights_ptr = weight_it->second.int8_ptr;
        }
    }

    // Handle the layer bias vector pointer mapping
    if (step.inputs.size() >= 3)
    {
        auto bias_it = try_find_tensor(step.inputs[2]);
        if (bias_it != weight_registry.end())
        {
            // Cast raw mapped memory straight into standard 32-bit integer array space
            step.bias_ptr = reinterpret_cast<const int32_t *>(bias_it->second.int8_ptr);
        }
    }

    // Set layout parameters
    step.input_byte_offset = accumulated_offset;
    step.custom_stride_step = stride_step;

    // --- BARE TOKEN ATTRIBUTE PARSING ---
    // Pass clean strings WITHOUT colons or quotes so the flexible scanner functions reliably
    step.axis = parse_json_scalar_int(node_context, "axis", 0, 0);
    step.ceil_mode = parse_json_scalar_int(node_context, "ceil_mode", 0, 0);
    step.cubic_coeff_a = parse_json_scalar_float(node_context, "cubic_coeff_a", -0.75f);
    step.dilations = parse_json_int_array(node_context, "dilations");
    step.group = parse_json_scalar_int(node_context, "group", 0, 1);
    step.strides = parse_json_int_array(node_context, "strides");
    step.pads = parse_json_int_array(node_context, "pads");
    step.kernel_shape = parse_json_int_array(node_context, "kernel_shape");

    return step;
}

bool InferencePipeline::export_optimized_pipeline(const std::string &output_json_path,
                                                  const std::vector<ExecutionStep> &pipeline)
{
    std::ofstream out_file(output_json_path);
    if (!out_file.is_open())
    {
        std::cerr << "Failed to open export target path: " << output_json_path << "\n";
        return false;
    }

    // Helper lambda to format integer arrays (strides, pads, shapes) into valid JSON strings
    auto format_int_array = [](const std::vector<int64_t> &vec) -> std::string
    {
        if (vec.empty())
            return "[]";
        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < vec.size(); ++i)
        {
            ss << vec[i] << (i < vec.size() - 1 ? ", " : "");
        }
        ss << "]";
        return ss.str();
    };

    // Helper lambda to format string arrays (inputs, outputs) into valid JSON string arrays
    auto format_string_array = [](const std::vector<std::string> &vec) -> std::string
    {
        if (vec.empty())
            return "[]";
        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < vec.size(); ++i)
        {
            ss << "\"" << vec[i] << "\"" << (i < vec.size() - 1 ? ", " : "");
        }
        ss << "]";
        return ss.str();
    };

    // Helper lambda to convert memory pointers into a structural hexadecimal string format
    auto format_pointer = [](const void *ptr) -> std::string
    {
        if (!ptr)
            return "\"nullptr\"";
        std::stringstream ss;
        ss << "\"0x" << std::hex << reinterpret_cast<uintptr_t>(ptr) << "\"";
        return ss.str();
    };

    // Open top-level layout block
    out_file << "{\n  \"optimized_execution_pipeline\": [\n";

    for (size_t i = 0; i < pipeline.size(); ++i)
    {
        const auto &step = pipeline[i];

        out_file << "    {\n";
        out_file << "      \"execution_order\": " << step.execution_order << ",\n";
        out_file << "      \"op_type\": \"" << step.op_type << "\",\n";
        out_file << "      \"node_name\": \"" << step.node_name << "\",\n";
        out_file << "      \"inputs\": " << format_string_array(step.inputs) << ",\n";
        out_file << "      \"outputs\": " << format_string_array(step.outputs) << ",\n";

        // Block 1: Structural Attributes
        out_file << "      \"attributes\": {\n";
        out_file << "        \"axis\": " << step.axis << ",\n";
        out_file << "        \"ceil_mode\": " << step.ceil_mode << ",\n";
        out_file << "        \"cubic_coeff_a\": " << step.cubic_coeff_a << ",\n";
        out_file << "        \"dilations\": " << format_int_array(step.dilations) << ",\n";
        out_file << "        \"group\": " << step.group << ",\n";
        out_file << "        \"strides\": " << format_int_array(step.strides) << ",\n";
        out_file << "        \"pads\": " << format_int_array(step.pads) << ",\n";
        out_file << "        \"kernel_shape\": " << format_int_array(step.kernel_shape) << "\n";
        out_file << "      },\n";

        // Block 2: Memory Addresses (Mapped via physical mmap offsets)
        out_file << "      \"memory_pointers\": {\n";
        out_file << "        \"weights_ptr\": " << format_pointer(step.weights_ptr) << ",\n";
        out_file << "        \"bias_ptr\": " << format_pointer(step.bias_ptr) << "\n";
        out_file << "      },\n";

        // Block 3: Fused Quantization Parameters
        out_file << "      \"quantization_parameters\": {\n";
        out_file << "        \"input_scale\": " << std::scientific << std::setprecision(8) << step.input_scale << ",\n";
        out_file << "        \"weight_scale\": " << step.weight_scale << ",\n";
        out_file << "        \"output_scale\": " << step.output_scale << ",\n";
        out_file << "        \"input_zero_point\": " << std::fixed << step.input_zero_point << ",\n";
        out_file << "        \"weight_zero_point\": " << step.weight_zero_point << ",\n";
        out_file << "        \"output_zero_point\": " << step.output_zero_point << "\n";
        out_file << "      },\n";

        // Block 4: Structural Trackers
        out_file << "      \"view_trackers\": {\n";
        out_file << "        \"input_byte_offset\": " << step.input_byte_offset << ",\n";
        out_file << "        \"custom_stride_step\": " << step.custom_stride_step << "\n";
        out_file << "      }\n";

        // Handle trailing trailing commas for array alignment elements
        out_file << "    }" << (i < pipeline.size() - 1 ? ",\n" : "\n");
    }

    out_file << "  ]\n}\n";
    out_file.close();
    return true;
}

void InferencePipeline::verify_compiled_memory_bounds(const std::vector<ExecutionStep> &pipeline)
{
    std::cout << "=== RUNNING MEMORY INTEGRITY BUFFER AUDIT ===\n";
    uintptr_t base_addr = reinterpret_cast<uintptr_t>(this->get_base_address());
    size_t buffer_limit = this->get_buffer_size();

    for (const auto &step : pipeline)
    {
        if (step.op_type == "Conv")
        {
            // Verify Weights Pointer Bounds
            if (step.weights_ptr != nullptr)
            {
                uintptr_t target_ptr = reinterpret_cast<uintptr_t>(step.weights_ptr);
                if (target_ptr < base_addr || target_ptr >= (base_addr + buffer_limit))
                {
                    std::cerr << "[FATAL] Node " << step.node_name
                              << " weights_ptr (" << step.weights_ptr
                              << ") points completely outside the mmap bounds!\n";
                }
            }

            // Verify Bias Pointer Bounds
            if (step.bias_ptr != nullptr)
            {
                uintptr_t target_ptr = reinterpret_cast<uintptr_t>(step.bias_ptr);
                if (target_ptr < base_addr || target_ptr >= (base_addr + buffer_limit))
                {
                    std::cerr << "[FATAL] Node " << step.node_name
                              << " bias_ptr (" << step.bias_ptr
                              << ") points completely outside the mmap bounds!\n";
                }
            }
        }
    }
    std::cout << "=== MEMORY INTEGRITY BUFFER AUDIT COMPLETE ===\n";
}
