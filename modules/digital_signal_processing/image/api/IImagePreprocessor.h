#ifndef DSP_IMAGE_I_IMAGE_PREPROCESSOR_H_
#define DSP_IMAGE_I_IMAGE_PREPROCESSOR_H_

#include <cstdint>
#include "dip_data_types.h"

namespace dsp
{
    namespace image
    {

        /**
         * @class IImagePreprocessor
         * @brief Abstract interface defining geometric image resizing and canvas transformations.
         */
        class IImagePreprocessor
        {
        public:
            virtual ~IImagePreprocessor() = default;

            /**
             * @brief Executes a geometric resize and layout adjustment pass on an incoming video frame.
             *
             * @param[in]  input_frame   The raw, unmanaged source frame data straight from the sensor.
             * @param[in]  target_width  The requested output image canvas width.
             * @param[in]  target_height The requested output image canvas height.
             * @param[out] out_buffer    Pointer to a pre-allocated memory block where raw pixels are written.
             * @param[in]  buffer_size   The strict maximum byte capacity of the out_buffer.
             *
             * @return TransformMetadata A ledger recording the exact padding changes and scaling factors applied.
             */
            virtual TransformMetadata Execute(const VideoFrame &input_frame,
                                              int32_t target_width,
                                              int32_t target_height,
                                              uint8_t *out_buffer,
                                              size_t buffer_size) = 0;
        };

    } // namespace image
} // namespace dsp

#endif // DSP_IMAGE_I_IMAGE_PREPROCESSOR_H_
