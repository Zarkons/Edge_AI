#ifndef DSP_IMAGE_I_TENSOR_PACKER_H_
#define DSP_IMAGE_I_TENSOR_PACKER_H_

#include <cstddef>
#include <cstdint>

namespace dsp
{
    namespace image
    {

        /**
         * @class ITensorPacker
         * @brief Abstract interface defining image-to-tensor structural transformation routines.
         *
         * Implementations of this interface handle channel permutation (HWC to CHW),
         * color-space re-ordering (e.g., BGR to RGB), and pixel value normalization
         * within deterministic, real-time performance bounds.
         */
        class ITensorPacker
        {
        public:
            /**
             * @brief Virtual destructor ensuring clean polymorphic cleanup of derived backends.
             */
            virtual ~ITensorPacker() = default;

            /**
             * @brief Transposes and normalizes an interleaved byte array into a flat contiguous float tensor array.
             *
             * @param[in]  preprocessed_img Pointer to the contiguous, letterboxed raw pixel byte array buffer (HWC).
             * @param[in]  width            The horizontal pixel width of the input image canvas area (e.g., 640).
             * @param[in]  height           The vertical pixel height of the input image canvas area (e.g., 640).
             * @param[in]  channels         The number of color channels present in the pixel source data (typically 3).
             * @param[out] out_tensor       Pointer to the pre-allocated float memory block where planar CHW data is written.
             * @param[in]  tensor_size      The strict maximum element capacity (float count) of the out_tensor block.
             *
             * @note This function must implement zero dynamic heap allocations to remain safe for QNX RTOS threads.
             */
            virtual void Pack(const uint8_t *preprocessed_img,
                              int32_t width,
                              int32_t height,
                              int32_t channels,
                              float *out_tensor,
                              size_t tensor_size) = 0;
        };

    } // namespace image
} // namespace dsp

#endif // DSP_IMAGE_I_TENSOR_PACKER_H_
