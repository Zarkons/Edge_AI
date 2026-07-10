#include "PlanarScaleTensorPacker.h"

namespace dsp
{
    namespace image
    {

        PlanarScaleTensorPacker::PlanarScaleTensorPacker() {}

        void PlanarScaleTensorPacker::Pack(const uint8_t *preprocessed_img,
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

            // Set up our planar tensor boundaries
            float *r_plane = out_tensor;
            float *g_plane = out_tensor + plane_stride;
            float *b_plane = out_tensor + (2 * plane_stride);

            for (size_t i = 0; i < plane_stride; ++i)
            {
                size_t hwc_base = i * channels;

                uint8_t b_val = preprocessed_img[hwc_base + 0];
                uint8_t g_val = preprocessed_img[hwc_base + 1];
                uint8_t r_val = preprocessed_img[hwc_base + 2];

                r_plane[i] = static_cast<float>(r_val) / 255.0f;
                g_plane[i] = static_cast<float>(g_val) / 255.0f;
                b_plane[i] = static_cast<float>(b_val) / 255.0f;
            }
        }

    } // namespace image
} // namespace dsp
