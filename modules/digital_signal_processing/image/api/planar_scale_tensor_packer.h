#ifndef DSP_IMAGE_PLANAR_SCALE_TENSOR_PACKER_H_
#define DSP_IMAGE_PLANAR_SCALE_TENSOR_PACKER_H_

#include <cstddef>
#include <cstdint>

namespace dsp
{
    namespace image
    {
        /**
         * @class PlanarScaleTensorPacker
         * @brief High-performance CPU implementation for transposing and normalizing image byte structures.
         *        Designed for zero-overhead static polymorphism. Defined entirely in the header
         *        to support complete cross-compilation inlining and SIMD auto-vectorization.
         */
        class PlanarScaleTensorPacker
        {
        public:
            /**
             * @brief Constructor initializing the packer. Implemented in header to preserve static layout.
             */
            explicit PlanarScaleTensorPacker() = default;

            /**
             * @brief Packs and normalizes interleaved bytes into flat contiguous float planes.
             *        Removed 'override' and virtual dependency to enable zero-cost static dispatch.
             */
            inline void Pack(const uint8_t *preprocessed_img,
                             int32_t width,
                             int32_t height,
                             int32_t channels,
                             float *out_tensor,
                             size_t tensor_size)
            {
                size_t required_elements = static_cast<size_t>(width) * height * channels;
                if (out_tensor == nullptr || preprocessed_img == nullptr || tensor_size < required_elements)
                {
                    return;
                }

                size_t plane_stride = static_cast<size_t>(width) * height;

                float *r_plane = out_tensor;
                float *g_plane = out_tensor + plane_stride;
                float *b_plane = out_tensor + (2 * plane_stride);

                // Multiplication is significantly faster than division on modern CPU math pipelines
                constexpr float inv_255 = 1.0f / 255.0f;

                for (size_t i = 0; i < plane_stride; ++i)
                {
                    size_t hwc_base = i * channels;

                    uint8_t b_val = preprocessed_img[hwc_base + 0];
                    uint8_t g_val = preprocessed_img[hwc_base + 1];
                    uint8_t r_val = preprocessed_img[hwc_base + 2];

                    r_plane[i] = static_cast<float>(r_val) * inv_255;
                    g_plane[i] = static_cast<float>(g_val) * inv_255;
                    b_plane[i] = static_cast<float>(b_val) * inv_255;
                }
            }
        };

    } // namespace image
} // namespace dsp

#endif // DSP_IMAGE_PLANAR_SCALE_TENSOR_PACKER_H_
