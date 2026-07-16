#ifndef DSP_IMAGE_IMAGE_PREPROCESSOR_CONCEPT_H_
#define DSP_IMAGE_IMAGE_PREPROCESSOR_CONCEPT_H_

#include <cstdint>
#include <concepts>
#include "dip_data_types.h"

namespace dsp
{
    namespace image
    {
        /**
         * @brief Compile-time concept enforcing the Image Preprocessor interface.
         */
        template <typename T>
        concept ImagePreprocessor = requires(T preprocessor,
                                             const VideoFrame &input_frame,
                                             int32_t target_width,
                                             int32_t target_height,
                                             uint8_t *out_buffer,
                                             size_t buffer_size) {
            { preprocessor.Execute(input_frame, target_width, target_height, out_buffer, buffer_size) }
              -> std::same_as<TransformMetadata>;
        };

    } // namespace image
} // namespace dsp

#endif // DSP_IMAGE_IMAGE_PREPROCESSOR_CONCEPT_H_
