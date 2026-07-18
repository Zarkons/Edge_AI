#ifndef DSP_IMAGE_TENSOR_PACKER_CONCEPT_H_
#define DSP_IMAGE_TENSOR_PACKER_CONCEPT_H_

#include <cstdint>
#include <cstddef>
#include <concepts>

namespace dsp::image
{
    /**
     * @brief Compile-time structural constraint for structural tensor normalization packers.
     */
    template <typename T>
    concept TensorPacker = requires(
        T packer,
        const uint8_t *preprocessed_img,
        int32_t width,
        int32_t height,
        int32_t channels,
        float *out_tensor,
        size_t tensor_size) {
        { packer.Pack(preprocessed_img, width, height, channels, out_tensor, tensor_size) } -> std::same_as<void>;
    };

} // namespace dsp::image

#endif // DSP_IMAGE_TENSOR_PACKER_CONCEPT_H_
