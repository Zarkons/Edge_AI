#ifndef DSP_IMAGE_DIP_DATA_TYPES_H_
#define DSP_IMAGE_DIP_DATA_TYPES_H_

#include <cstdint>

/**
 * @file dip_data_types.h
 * @brief Defines the fundamental data structures for the dsp::image processing subsystem.
 */
namespace dsp
{
    namespace image
    {

        /**
         * @struct VideoFrame
         * @brief A lightweight, zero-copy wrapper around a raw camera image buffer.
         *
         * This structure does not own or allocate memory. It wraps a raw pointer pointing
         * directly to the physical hardware buffer to eliminate memory copies within the
         * real-time execution loop.
         */
        struct VideoFrame
        {
            const uint8_t *data_ptr = nullptr; /**< Pointer to the contiguous pixel byte array in memory. */
            int32_t width = 0;                 /**< Width of the image in pixels. */
            int32_t height = 0;                /**< Height of the image in pixels. */
            int32_t channels = 0;              /**< Number of color channels (e.g., 3 for BGR/RGB). */
            int32_t stride = 0;                /**< Total row width (stride) in bytes, including hardware padding. */
        };

        /**
         * @struct TransformMetadata
         * @brief Tracks geometric modifications to enable accurate bounding box coordinate re-mapping.
         *
         * Stores the padding offsets and scaling constants generated during the letterbox
         * resizing pass (e.g., transforming a 1080p frame to 640x640), allowing output bounding
         * boxes to be inversely mapped back onto the original frame resolution.
         */
        struct TransformMetadata
        {
            int32_t src_width = 0;     /**< Original width of the incoming camera frame. */
            int32_t src_height = 0;    /**< Original height of the incoming camera frame. */
            int32_t dst_width = 0;     /**< Target image width required by the model (640). */
            int32_t dst_height = 0;    /**< Target image height required by the model (640). */
            int32_t pad_x = 0;         /**< Horizontal symmetric padding (gray border width) in pixels. */
            int32_t pad_y = 0;         /**< Vertical symmetric padding (gray border height) in pixels. */
            float scale_factor = 1.0f; /**< The scaling factor applied to maintain aspect ratio. */
        };

    } // namespace image
} // namespace dsp

#endif // DSP_IMAGE_DIP_DATA_TYPES_H_
