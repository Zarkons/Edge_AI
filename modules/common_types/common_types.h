#ifndef ML_TYPES_H_
#define ML_TYPES_H_

#include <vector>
#include <cstdint>

namespace QuantizationTypes
{
    // Aliases for TensorFlow Lite quantization parameters
    using ScaleType = float;
    using ZeroPointType = int8_t;
    using QuantizedInt8Type = int8_t;

    struct QuantizationParams
    {
        ScaleType scale = 1.0f;
        ZeroPointType zero_point = 0;
    };
}

namespace MlTypesDynamic
{
    // Global, domain-agnostic aliases for any edge inference pipeline
    using RawSignalFloatVec = std::vector<float>;
    using FeatureFloatVec = std::vector<float>;
    using FeatureQuantInt8Vec = std::vector<int8_t>;
    using PredictionFloatVec = std::vector<float>;
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
