#ifndef ML_TYPES_H_
#define ML_TYPES_H_

#include <vector>
#include <cstdint>

/**
 * @file common_types.h
 * @brief Common quantization and tensor array shapes for Edge AI layer arrays.
 */

/**
 * @brief The quantization_types namespace contains type aliases and structures for quantization parameters used in machine learning pipelines.
 */
namespace quantization_types
{
    /**
     * @brief Type alias for the scale parameter used in quantization. Scale is used to map the range of floating-point values to the range of quantized integer values.
     */
    using ScaleType = float;
    /**
     * @brief Type alias for the zero-point parameter used in quantization. Zero-point is used to shift the range of quantized integer values to align with the range of floating-point values.
     */
    using ZeroPointType = int8_t;
    /**
     * @brief Type alias for the quantized int8 type used in quantization. This type represents the quantized values after applying scale and zero-point.
     */
    using QuantizedInt8Type = int8_t;

    /**
     * @brief Structure to hold quantization parameters, including scale and zero-point.
     */
    struct QuantizationParams
    {
        ScaleType scale = 1.0f;
        ZeroPointType zero_point = 0;
    };
}

/**
 * @brief The ml_types_dynamic namespace contains type aliases for dynamic-sized containers used in machine learning pipelines.
 */
namespace ml_types_dynamic
{
    /**
     * @brief Type alias for dynamic-sized container used to represent raw audio signal.
     */
    using RawSignalFloatVec = std::vector<float>;

    /**
     * @brief Type alias for dynamic-sized container used to represent features extracted from raw audio signal.
     */
    using FeatureFloatVec = std::vector<float>;

    /**
     * @brief Type alias for dynamic-sized container used to represent quantized features extracted from raw audio signal.
     */
    using FeatureQuantInt8Vec = std::vector<int8_t>;

    /**
     * @brief Type alias for dynamic-sized container used to represent predictions output from the model.
     */
    using PredictionFloatVec = std::vector<float>;

    /**
     * @brief Type alias for dynamic-sized container used to represent quantized predictions output from the model.
     */
    using PredictionQuantInt8Vec = std::vector<int8_t>;
}

// ToDo: Consider adding a static version of the above types for compile-time fixed-size arrays, if needed in the future
/* namespace MlEngineStatic
{
    // Template aliases: The 'Size' parameter is provided during instantiation
    template <size_t Size>
    using RawSignalFloatArr = std::array<float, Size>;

    template <size_t Size>
    using FeatureFloatArr = std::array<float, Size>;

    template <size_t Size>
    using FeatureQuantInt8Arr = std::array<int8_t, Size>;

    template <size_t Size>
    using PredictionFloatArr = std::array<float, Size>;

    template <size_t Size>
    using PredictionQuantInt8Arr = std::array<int8_t, Size>;
} */

#endif // ML_TYPES_H_
