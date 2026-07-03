#include "CPULetterboxPreprocessor.h"
#include <algorithm>
#include <cmath>

namespace dsp
{
    namespace image
    {

        TransformMetadata CPULetterboxPreprocessor::Execute(const VideoFrame &input_frame,
                                                            int32_t target_width,
                                                            int32_t target_height,
                                                            uint8_t *out_buffer,
                                                            size_t buffer_size)
        {
            TransformMetadata meta;
            meta.src_width = input_frame.width;
            meta.src_height = input_frame.height;
            meta.dst_width = target_width;
            meta.dst_height = target_height;

            size_t required_bytes = static_cast<size_t>(target_width) * target_height * input_frame.channels;
            if (buffer_size < required_bytes || out_buffer == nullptr)
            {
                return meta;
            }

            // Compute aspect scaling ratios
            float ratio_w = static_cast<float>(target_width) / input_frame.width;
            float ratio_h = static_cast<float>(target_height) / input_frame.height;
            meta.scale_factor = std::min(ratio_w, ratio_h);

            int32_t unpad_w = static_cast<int32_t>(std::round(input_frame.width * meta.scale_factor));
            int32_t unpad_h = static_cast<int32_t>(std::round(input_frame.height * meta.scale_factor));

            meta.pad_x = (target_width - unpad_w) / 2;
            meta.pad_y = (target_height - unpad_h) / 2;

            // Fill output canvas canvas background with gray constant (114)
            std::fill(out_buffer, out_buffer + required_bytes, static_cast<uint8_t>(114));

            for (int32_t y = 0; y < unpad_h; ++y)
            {
                float src_y = static_cast<float>(y) / meta.scale_factor;
                int32_t y_low = static_cast<int32_t>(std::floor(src_y));
                int32_t y_high = std::min(y_low + 1, input_frame.height - 1);
                float y_weight = src_y - y_low;

                int32_t target_y = y + meta.pad_y;
                size_t dest_row_offset = static_cast<size_t>(target_y) * target_width * input_frame.channels;

                // Skip input row padding using hardware stride metrics
                const uint8_t *row_low_ptr = input_frame.data_ptr + (y_low * input_frame.stride);
                const uint8_t *row_high_ptr = input_frame.data_ptr + (y_high * input_frame.stride);

                for (int32_t x = 0; x < unpad_w; ++x)
                {
                    float src_x = static_cast<float>(x) / meta.scale_factor;
                    int32_t x_low = static_cast<int32_t>(std::floor(src_x));
                    int32_t x_high = std::min(x_low + 1, input_frame.width - 1);
                    float x_weight = src_x - x_low;

                    int32_t target_x = x + meta.pad_x;
                    size_t dest_pixel_idx = dest_row_offset + (static_cast<size_t>(target_x) * input_frame.channels);

                    int32_t xl_offset = x_low * input_frame.channels;
                    int32_t xh_offset = x_high * input_frame.channels;

                    for (int32_t c = 0; c < input_frame.channels; ++c)
                    {
                        float p00 = row_low_ptr[xl_offset + c];
                        float p10 = row_low_ptr[xh_offset + c];
                        float p01 = row_high_ptr[xl_offset + c];
                        float p11 = row_high_ptr[xh_offset + c];

                        float top = p00 + x_weight * (p10 - p00);
                        float bottom = p01 + x_weight * (p11 - p01);
                        float interpolated_val = top + y_weight * (bottom - top);

                        out_buffer[dest_pixel_idx + c] = static_cast<uint8_t>(interpolated_val);
                    }
                }
            }
            return meta;
        }

    } // namespace image
} // namespace dsp
