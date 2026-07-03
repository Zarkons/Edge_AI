#ifndef DSP_IMAGE_CPU_LETTERBOX_PREPROCESSOR_H_
#define DSP_IMAGE_CPU_LETTERBOX_PREPROCESSOR_H_

#include <cstddef>
#include <cstdint>
#include "dip_data_types.h"
#include "IImagePreprocessor.h"

namespace dsp
{
    namespace image
    {

        /**
         * @class CPULetterboxPreprocessor
         * @brief CPU-bound implementation of a bilinear resizing and letterbox padding pipeline.
         */
        class CPULetterboxPreprocessor : public IImagePreprocessor
        {
        public:
            CPULetterboxPreprocessor() = default;
            virtual ~CPULetterboxPreprocessor() = default;

            TransformMetadata Execute(const VideoFrame &input_frame,
                                      int32_t target_width,
                                      int32_t target_height,
                                      uint8_t *out_buffer,
                                      size_t buffer_size) override;
        };

    } // namespace image
} // namespace dsp

#endif // DSP_IMAGE_CPU_LETTERBOX_PREPROCESSOR_H_
