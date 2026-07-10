#ifndef DSP_IMAGE_PLANAR_SCALE_TENSOR_PACKER_H_
#define DSP_IMAGE_PLANAR_SCALE_TENSOR_PACKER_H_

#include <cstddef>
#include <cstdint>
#include "ITensorPacker.h"

namespace dsp
{
    namespace image
    {

        /**
         * @class PlanarScaleTensorPacker
         * @brief High-performance CPU implementation for transposing and normalizing image byte structures.
         *
         * Maps interleaved HWC image arrays directly into normalized planar CHW float arrays
         * without executing any runtime heap modifications.
         */
        class PlanarScaleTensorPacker : public ITensorPacker
        {
        public:
            /**
             * @brief Constructor initializing the packer with behavioral configuration parameters.
             */
            explicit PlanarScaleTensorPacker();

            virtual ~PlanarScaleTensorPacker() = default;

            /**
             * @brief Packs and normalizes interleaved bytes into flat contiguous float planes.
             */
            void Pack(const uint8_t *preprocessed_img,
                      int32_t width,
                      int32_t height,
                      int32_t channels,
                      float *out_tensor,
                      size_t tensor_size) override;

        private:
        };

    } // namespace image
} // namespace dsp

#endif // DSP_IMAGE_PLANAR_SCALE_TENSOR_PACKER_H_
