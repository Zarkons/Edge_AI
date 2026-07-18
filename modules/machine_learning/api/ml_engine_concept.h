#ifndef ML_ENGINE_CONCEPT_H_
#define ML_ENGINE_CONCEPT_H_

#include <cstdint>
#include <cstddef>
#include <concepts>
#include "ml_types.h"

namespace ml
{
    namespace engine
    {
        /**
         * @brief Compile-time structural contract for AI Inference Engines.
         */
        template <typename T>
        concept InferenceEngine = requires(
            T engine,
            int32_t intra_threads,
            int32_t inter_threads,
            const std::string &model_path,
            const float *input_tensor,
            const int64_t *input_shape,
            size_t shape_dims,
            float *out_preallocated_buffer,
            size_t out_buffer_capacity,
            InferenceOutput &out_result) {
            { engine.Initialize(model_path, intra_threads, inter_threads) } -> std::same_as<bool>;

            { engine.RunInference(input_tensor, input_shape, shape_dims,
                                  out_preallocated_buffer, out_buffer_capacity, out_result) }
              -> std::same_as<bool>;
            { engine.GetClassNames() } -> std::same_as<std::vector<std::string>>;
        };
    } // namespace engine
} // namespace ml

#endif // ML_ENGINE_CONCEPT_H_
