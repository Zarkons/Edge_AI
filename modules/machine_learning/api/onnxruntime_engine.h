#ifndef ONNXRUNTIME_ENGINE_H
#define ONNXRUNTIME_ENGINE_H

#include <string>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include "ml_types.h"

namespace ml
{
    namespace engine
    {
        /**
         * @class ONNXRuntimeEngine
         * @brief A wrapper around the ONNX Runtime C++ API for model inference.
         *
         * This class provides a clean and straightforward interface for initializing
         * an ONNX model, running inference, and retrieving class names.
         */
        class ONNXRuntimeEngine
        {
        public:
            ONNXRuntimeEngine() = default;

            /**
             * @brief Initializes the ONNX Runtime engine with the specified model path and threading options.
             *
             * @param model_path The file path to the ONNX model.
             * @param intra_op_threads The number of threads to use for intra-op parallelism.
             * @param inter_op_threads The number of threads to use for inter-op parallelism.
             * @return true if initialization is successful, false otherwise.
             */
            bool Initialize(const std::string &model_path,
                            int32_t intra_op_threads,
                            int32_t inter_op_threads);

            /**
             * @brief Runs inference on the provided input tensor and shape, storing results in a preallocated output buffer.
             *
             * @param input_tensor Pointer to the input tensor data.
             * @param input_shape Pointer to the shape of the input tensor.
             * @param shape_dims The number of dimensions in the input shape.
             * @param out_preallocated_buffer Pointer to a preallocated buffer for output data.
             * @param out_buffer_capacity The capacity of the preallocated output buffer.
             * @param out_result Reference to an InferenceOutput structure to store the results.
             * @return true if inference is successful, false otherwise.
             */
            bool RunInference(const float *input_tensor,
                              const int64_t *input_shape,
                              size_t shape_dims,
                              float *out_preallocated_buffer,
                              size_t out_buffer_capacity,
                              InferenceOutput &out_result);

            /**
             * @brief Retrieves the class names associated with the model's output.
             * @return A vector of class names as strings.
             */
            std::vector<std::string> GetClassNames();

        private:
            std::unique_ptr<Ort::Env> m_env;
            std::unique_ptr<Ort::SessionOptions> m_session_options;
            std::unique_ptr<Ort::Session> m_session{nullptr};
            std::unique_ptr<Ort::IoBinding> m_io_binding;
            std::string m_input_name;
            std::string m_output_name;
            ONNXTensorElementDataType m_input_elem_type{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
            Ort::MemoryInfo m_memory_info{nullptr};
            size_t m_input_tensor_size{0};
            size_t m_output_element_count{0};
            size_t m_output_shape_dims{0};
            int64_t m_output_shape[8]{};
        };
    } // namespace engine
} // namespace ml

#endif // ONNXRUNTIME_ENGINE_H
