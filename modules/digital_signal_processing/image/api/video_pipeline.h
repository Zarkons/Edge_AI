#ifndef DSP_IMAGE_VIDEO_PIPELINE_H_
#define DSP_IMAGE_VIDEO_PIPELINE_H_

#include <utility>
#include <cstdint>
#include <cstddef>
#include "dip_data_types.h"
#include "image_preprocessor_concept.h"
#include "tensor_packer_concept.h"
#include "ml_engine_concept.h"
#include "ml_types.h"

namespace dsp
{
    namespace image
    {
        /**
         * @class VideoPipeline
         * @brief Template-driven orchestration engine for streaming video frames.
         *        Fully decoupled across Preprocessing, Tensor Packing, and AI Execution blocks.
         */
        template <
            dsp::image::ImagePreprocessor TPreprocessor,
            dsp::image::TensorPacker TTensorPacker,
            ml::engine::InferenceEngine TEngine>
        class VideoPipeline
        {
        public:
            /**
             * @brief Constructs the master video pipeline using three externally managed buffer allocations.
             *
             * @param preprocessor Concrete resizing worker instance (moved by value).
             * @param packer Concrete tensor layout transformer instance (moved by value).
             * @param engine Concrete AI inference driver instance (moved by value).
             * @param target_width Requested canvas processing width (e.g., 640).
             * @param target_height Requested canvas processing height (e.g., 640).
             * @param target_channels Dynamic image color channel validation metric (e.g., 3).
             * @param preprocess_buffer Memory location where raw byte pixels are scaled and padded.
             * @param preprocess_capacity Strict capacity boundary of the raw byte buffer.
             * @param inference_buffer Memory location where formatted planar floats are packed.
             * @param inference_capacity Strict capacity element count boundary of the float tensor.
             * @param output_buffer Memory location where raw network layer predictions are written.
             * @param output_capacity Strict capacity boundary of the hardware output array.
             */
            explicit VideoPipeline(TPreprocessor preprocessor,
                                   TTensorPacker packer,
                                   TEngine engine,
                                   int32_t target_width,
                                   int32_t target_height,
                                   int32_t target_channels,
                                   uint8_t *preprocess_buffer,
                                   size_t preprocess_capacity,
                                   float *inference_buffer,
                                   size_t inference_capacity,
                                   int64_t *input_shape_buffer,
                                   size_t input_shape_capacity,
                                   float *output_buffer,
                                   size_t output_capacity)
                : preprocessor_(std::move(preprocessor)),
                  packer_(std::move(packer)),
                  engine_(std::move(engine)),
                  target_width_(target_width),
                  target_height_(target_height),
                  target_channels_(target_channels),
                  prep_buffer_(preprocess_buffer),
                  prep_capacity_(preprocess_capacity),
                  inf_buffer_(inference_buffer),
                  inf_capacity_(inference_capacity),
                  input_shape_buffer_(input_shape_buffer),
                  input_shape_capacity_(input_shape_capacity),
                  out_buffer_(output_buffer),
                  out_capacity_(output_capacity)
            {
                // Guarantee zero runtime heap allocations by binding our non-owning struct
                // directly to localized, stack-friendly static array blocks
                out_result_.shape = out_shape_storage_;
                out_result_.shape_capacity = 8;
            }

            // Rule of 5 lifecycle handling
            ~VideoPipeline() = default;
            VideoPipeline(const VideoPipeline &) = delete;
            VideoPipeline &operator=(const VideoPipeline &) = delete;
            VideoPipeline(VideoPipeline &&) noexcept = default;
            VideoPipeline &operator=(VideoPipeline &&) noexcept = default;

            /**
             * @brief Handles geometric preprocessing and structural tensor layout transformation.
             * @param raw_frame The raw incoming frame straight from the hardware video sensor.
             * @return true if both preprocessing and layout packing succeed; false otherwise.
             */
            bool StreamFramePreprocess(const VideoFrame &raw_frame)
            {
                if (raw_frame.channels != target_channels_ ||
                    prep_buffer_ == nullptr ||
                    inf_buffer_ == nullptr)
                {
                    return false;
                }

                latest_meta_ = preprocessor_.Execute(
                    raw_frame,
                    target_width_,
                    target_height_,
                    prep_buffer_,
                    prep_capacity_);

                packer_.Pack(
                    prep_buffer_,
                    target_width_,
                    target_height_,
                    target_channels_,
                    inf_buffer_,
                    inf_capacity_);

                return true;
            }

            /**
             * @brief Maps the input tensor dimensions and executes the forward inference pass.
             *        Expects that StreamFramePreprocess has already been called successfully for the current frame.
             * @return true if the neural network execution succeeds; false otherwise.
             */
            bool InferenceExecute()
            {
                // Safety Guard: Verify output destination buffer pointer safety
                if (out_buffer_ == nullptr || inf_buffer_ == nullptr)
                {
                    return false;
                }

                // Stage 4: High-Performance Forward Execution Pass to the neural network backend
                return engine_.RunInference(
                    inf_buffer_,
                    input_shape_buffer_,
                    input_shape_capacity_,
                    out_buffer_,
                    out_capacity_,
                    out_result_);
            }

            /**
             * @brief Returns a read-only reference to the latest structures execution output layer.
             */
            const InferenceOutput &GetLatestResult() const
            {
                return out_result_;
            }

            /**
             * @brief Returns preprocessing transform metadata for the latest processed frame.
             */
            const TransformMetadata &GetLatestTransformMetadata() const
            {
                return latest_meta_;
            }

            bool InitializeEngine(const std::string &model_path, int32_t intra_threads, int32_t inter_threads)
            {
                return engine_.Initialize(model_path, intra_threads, inter_threads);
            }

            std::vector<std::string> GetClassNames()
            {
                return engine_.GetClassNames();
            }

        private:
            TPreprocessor preprocessor_; ///< Resizing worker embedded by value
            TTensorPacker packer_;       ///< Color/Permutation layout worker embedded by value
            TEngine engine_;             ///< Hardware AI execution driver embedded by value

            int32_t target_width_;
            int32_t target_height_;
            int32_t target_channels_;

            uint8_t *prep_buffer_; ///< Non-owning tracking pointer for resized bytes
            size_t prep_capacity_;

            int64_t *input_shape_buffer_; ///< Non-owning tracking pointer for input tensor shape
            size_t input_shape_capacity_;

            float *inf_buffer_; ///< Non-owning tracking pointer for input float tensors
            size_t inf_capacity_;

            float *out_buffer_; ///< Non-owning tracking pointer for engine prediction dumps
            size_t out_capacity_;

            InferenceOutput out_result_;
            TransformMetadata latest_meta_{};
            int64_t out_shape_storage_[8]{0}; ///< 64-byte localized stack block preventing heap allocation
        };

    } // namespace image
} // namespace dsp

#endif // DSP_IMAGE_VIDEO_PIPELINE_H_
