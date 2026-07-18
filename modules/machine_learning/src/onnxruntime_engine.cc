#include "onnxruntime_engine.h"
#include <algorithm>
#include <iostream>
#include <sstream>

namespace
{
    const char *OnnxTypeToString(ONNXType type)
    {
        switch (type)
        {
        case ONNX_TYPE_UNKNOWN:
            return "unknown";
        case ONNX_TYPE_TENSOR:
            return "tensor";
        case ONNX_TYPE_SEQUENCE:
            return "sequence";
        case ONNX_TYPE_MAP:
            return "map";
        case ONNX_TYPE_OPAQUE:
            return "opaque";
        case ONNX_TYPE_SPARSETENSOR:
            return "sparse_tensor";
        case ONNX_TYPE_OPTIONAL:
            return "optional";
        default:
            return "unrecognized";
        }
    }

    const char *TensorTypeToString(ONNXTensorElementDataType type)
    {
        switch (type)
        {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            return "undefined";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return "int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            return "uint16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            return "int16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            return "float64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return "bool";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            return "float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            return "bfloat16";
        default:
            return "unknown";
        }
    }

    std::string ShapeToString(const std::vector<int64_t> &shape)
    {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < shape.size(); ++i)
        {
            oss << shape[i];
            if (i + 1 < shape.size())
            {
                oss << "x";
            }
        }
        oss << "]";
        return oss.str();
    }

    std::string ShapeToString(const int64_t *shape, size_t dims)
    {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < dims; ++i)
        {
            oss << shape[i];
            if (i + 1 < dims)
            {
                oss << "x";
            }
        }
        oss << "]";
        return oss.str();
    }
}

namespace ml
{
    namespace engine
    {
        bool ONNXRuntimeEngine::Initialize(const std::string &model_path,
                                           int32_t intra_op_threads,
                                           int32_t inter_op_threads)
        {
            try
            {
                m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,
                                                   "ONNXRuntimeEngine_Inference");

                m_session_options = std::make_unique<Ort::SessionOptions>();
                m_session_options->SetIntraOpNumThreads(intra_op_threads);
                m_session_options->SetInterOpNumThreads(inter_op_threads);
                m_session_options->SetExecutionMode(inter_op_threads > 1 ? ExecutionMode::ORT_PARALLEL : ExecutionMode::ORT_SEQUENTIAL);
                m_session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

                // 3. Create the Session (Dereference m_env and m_session_options using *)
                m_session = std::make_unique<Ort::Session>(*m_env, model_path.c_str(), *m_session_options);

                Ort::AllocatorWithDefaultOptions m_allocator;

                m_input_name = m_session->GetInputNameAllocated(0, m_allocator).get();
                m_output_name = m_session->GetOutputNameAllocated(0, m_allocator).get();

                m_io_binding = std::make_unique<Ort::IoBinding>(*m_session);

                auto input_type_info = m_session->GetInputTypeInfo(0);
                auto output_type_info = m_session->GetOutputTypeInfo(0);
                ONNXType input_onnx_type = input_type_info.GetONNXType();
                ONNXType output_onnx_type = output_type_info.GetONNXType();

                std::cout << "[ONNX][Init] model_path=" << model_path << std::endl;
                std::cout << "[ONNX][Init] input name='" << m_input_name << "' onnx_type="
                          << OnnxTypeToString(input_onnx_type) << "(" << static_cast<int>(input_onnx_type) << ")";
                if (input_onnx_type == ONNX_TYPE_TENSOR)
                {
                    auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
                    ONNXTensorElementDataType elem_type = tensor_info.GetElementType();
                    m_input_elem_type = elem_type;
                    auto input_shape_vec = tensor_info.GetShape();
                    m_input_tensor_size = 1;
                    for (auto dim : input_shape_vec)
                    {
                        m_input_tensor_size *= static_cast<size_t>(dim);
                    }
                    std::cout << " elem_type=" << TensorTypeToString(elem_type)
                              << "(" << static_cast<int>(elem_type) << ")"
                              << " shape=" << ShapeToString(input_shape_vec);
                }
                std::cout << std::endl;

                std::cout << "[ONNX][Init] output name='" << m_output_name << "' onnx_type="
                          << OnnxTypeToString(output_onnx_type) << "(" << static_cast<int>(output_onnx_type) << ")";
                if (output_onnx_type == ONNX_TYPE_TENSOR)
                {
                    auto tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
                    ONNXTensorElementDataType elem_type = tensor_info.GetElementType();
                    auto output_shape_vec = tensor_info.GetShape();
                    m_output_element_count = tensor_info.GetElementCount();
                    m_output_shape_dims = output_shape_vec.size();
                    for (size_t i = 0; i < m_output_shape_dims; ++i)
                    {
                        m_output_shape[i] = output_shape_vec[i];
                    }
                    std::cout << " elem_type=" << TensorTypeToString(elem_type)
                              << "(" << static_cast<int>(elem_type) << ")"
                              << " shape=" << ShapeToString(output_shape_vec);
                }
                std::cout << std::endl;
                m_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                return true;
            }
            catch (const Ort::Exception &e)
            {
                std::cerr << "ONNX Engine Initialization Failed: " << e.what() << std::endl;
                return false;
            }
        }

        bool ONNXRuntimeEngine::RunInference(const float *input_tensor,
                                             const int64_t *input_shape,
                                             size_t shape_dims,
                                             float *out_preallocated_buffer,
                                             size_t out_buffer_capacity,
                                             InferenceOutput &out_result)
        {
            if (!m_session || !input_tensor || !input_shape || !out_preallocated_buffer || !out_result.shape || out_result.shape_capacity == 0) [[unlikely]]
            {
                return false;
            }

            // 1. Wrap preallocated input memory (Zero-Copy) — size cached at init
            Ort::Value input_onnx_tensor = Ort::Value::CreateTensor<float>(
                m_memory_info, const_cast<float *>(input_tensor), m_input_tensor_size, input_shape, shape_dims);

            // 2. Safety checks using shape metadata cached at init
            if (m_output_element_count > out_buffer_capacity) [[unlikely]]
            {
                std::cerr << "[ONNX][Run] Output buffer too small: required=" << m_output_element_count
                          << " capacity=" << out_buffer_capacity << std::endl;
                return false;
            }

            if (m_output_shape_dims > out_result.shape_capacity) [[unlikely]]
            {
                std::cerr << "[ONNX][Run] Output shape dims exceed target structure capacity" << std::endl;
                return false;
            }

            // 3. Wrap preallocated output memory (Zero-Copy) — shape cached at init
            Ort::Value output_onnx_tensor = Ort::Value::CreateTensor<float>(
                m_memory_info, out_preallocated_buffer, out_buffer_capacity, m_output_shape, m_output_shape_dims);

            // 4. IoBinding reset and assignation
            m_io_binding->ClearBoundInputs();
            m_io_binding->ClearBoundOutputs();
            m_io_binding->BindInput(m_input_name.c_str(), input_onnx_tensor);
            m_io_binding->BindOutput(m_output_name.c_str(), output_onnx_tensor);

            // 5. Ultra-fast inference pass
            m_session->Run(Ort::RunOptions{nullptr}, *m_io_binding);

            // 6. Populate result metadata — no copies, no allocations
            out_result.data_ptr = out_preallocated_buffer;
            out_result.element_count = m_output_element_count;
            out_result.shape_dims = m_output_shape_dims;

            for (size_t i = 0; i < m_output_shape_dims; ++i)
            {
                out_result.shape[i] = m_output_shape[i];
            }

            return true;
        }

        // Ensure this declaration signature is added to your ONNXRuntimeEngine header file!
        std::vector<std::string> ONNXRuntimeEngine::GetClassNames()
        {
            std::vector<std::string> class_names;

            try
            {
                // 1. Guard check to make sure the session pointer is allocated
                if (!m_session)
                {
                    std::cerr << "[WARN] Cannot get class names; ONNX Session is null." << std::endl;
                    return {};
                }

                // FIX: Replaced session. with m_session-> to handle the smart pointer layout
                Ort::ModelMetadata metadata = m_session->GetModelMetadata();
                Ort::AllocatorWithDefaultOptions allocator;

                // 2. Query the custom serialized "names" metadata key written by Ultralytics
                auto allocated_names = metadata.LookupCustomMetadataMapAllocated("names", allocator);

                if (allocated_names)
                {
                    std::string names_dict_str(allocated_names.get());

                    // Format check: Extract tags within matching single quotes safely
                    size_t pos = 0;
                    while ((pos = names_dict_str.find("'", pos)) != std::string::npos)
                    {
                        size_t end_pos = names_dict_str.find("'", pos + 1);
                        if (end_pos == std::string::npos)
                        {
                            break;
                        }

                        std::string name = names_dict_str.substr(pos + 1, end_pos - pos - 1);
                        class_names.push_back(name);
                        pos = end_pos + 1;
                    }
                }
            }
            catch (const std::exception &e)
            {
                std::cerr << "[WARN] ONNX Runtime exception during metadata extraction: " << e.what() << std::endl;
            }

            return class_names;
        }

    }
}
