#ifndef ONNXRUNTIME_ENGINE_H
#define ONNXRUNTIME_ENGINE_H

#include <string>
#include <memory>
#include <onnxruntime_cxx_api.h>

namespace onnxruntime
{
    namespace inference
    {
        struct InferenceOutput
        {
            float *data_ptr{nullptr};
            size_t element_count{0};
            int64_t *shape{nullptr};
            size_t shape_capacity{0};
            size_t shape_dims{0};
        };

        class ONNXRuntimeEngine
        {
        public:
            ONNXRuntimeEngine() = default;
            ~ONNXRuntimeEngine() = default;

            bool Initialize(const std::string &model_path,
                            int32_t intra_op_threads,
                            int32_t inter_op_threads);

            // Clean, straightforward C++ method. Zero compilation bloat.
            bool Run(const float *input_tensor,
                     const int64_t *input_shape,
                     size_t shape_dims,
                     float *out_preallocated_buffer,
                     size_t out_buffer_capacity,
                     InferenceOutput &out_result);

            std::vector<std::string> GetClassNames();

        private:
            Ort::Env m_env{ORT_LOGGING_LEVEL_WARNING,
                           "YOLOv8_Inference"};
            Ort::SessionOptions m_session_options;
            std::unique_ptr<Ort::Session> m_session{nullptr};
            std::string m_input_name;
            std::string m_output_name;
        };
    }
}

#endif // ONNXRUNTIME_ENGINE_H
