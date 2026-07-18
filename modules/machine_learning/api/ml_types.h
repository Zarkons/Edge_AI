#ifndef ML_TYPES_H_
#define ML_TYPES_H_

#include <cstdint>
#include <cstddef>

/**
 * @struct InferenceOutput
 * @brief Represents the output of an inference operation.
 *
 * This structure encapsulates the output data pointer, the number of elements,
 * the shape pointer, the shape capacity, and the number of shape dimensions.
 */
struct InferenceOutput
{
    float *data_ptr{nullptr};
    size_t element_count{0};
    int64_t *shape{nullptr};
    size_t shape_capacity{0};
    size_t shape_dims{0};
};

#endif // ML_TYPES_H_
