#ifndef OBJ_REC_APP_YOLOV8_POST_PROCESSOR_H_
#define OBJ_REC_APP_YOLOV8_POST_PROCESSOR_H_

#include <cstdint>
#include <vector>
#include "dip_data_types.h"

namespace obj_rec
{
    namespace app
    {
        /**
         * @struct Detection
         * @brief Unified structure representing a parsed, filtered object classification.
         */
        struct Detection
        {
            float x1 = 0.0f;
            float y1 = 0.0f;
            float x2 = 0.0f;
            float y2 = 0.0f;
            float confidence = 0.0f;
            int32_t class_id = -1;
        };

        class YOLOv8PostProcessor
        {
        public:
            YOLOv8PostProcessor() = default;
            ~YOLOv8PostProcessor() = default;

            /**
             * @brief Parses a flat raw YOLOv8 output tensor array into final object boundaries.
             *
             * @param[in]  raw_output     Pointer to the pre-allocated flat float results from ONNX Runtime.
             * @param[in]  conf_threshold Score floor cutoff limit (e.g., 0.25f).
             * @param[in]  iou_threshold  Overlap limits for suppression logic (e.g., 0.45f).
             * @param[in]  meta           Metadata ledger generated during the preprocessing resize step.
             *
             * @return std::vector<Detection> Cleanly filtered object boundaries mapped back to original image space.
             */
            std::vector<Detection> Execute(const float *raw_output,
                                           float conf_threshold,
                                           float iou_threshold,
                                           const dsp::image::TransformMetadata &meta);

        private:
            float ComputeIoU(const Detection &a, const Detection &b) const;
        };
    }
}

#endif // OBJ_REC_APP_YOLOV8_POST_PROCESSOR_H_
