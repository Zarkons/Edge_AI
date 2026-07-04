#include "YOLOv8PostProcessor.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <array>

namespace obj_rec
{
    namespace app
    {
        float YOLOv8PostProcessor::ComputeIoU(const Detection &a, const Detection &b) const
        {
            float x1 = std::max(a.x1, b.x1);
            float y1 = std::max(a.y1, b.y1);
            float x2 = std::min(a.x2, b.x2);
            float y2 = std::min(a.y2, b.y2);

            float intersection_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            if (intersection_area <= 0.0f)
            {
                return 0.0f;
            }

            float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
            float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);

            return intersection_area / (area_a + area_b - intersection_area);
        }

        std::vector<Detection> YOLOv8PostProcessor::Execute(const float *raw_output,
                                                            float conf_threshold,
                                                            float iou_threshold,
                                                            const dsp::image::TransformMetadata &meta)
        {
            constexpr int32_t num_channels = 84;     // 4 box geometry dimensions + 80 COCO classes
            constexpr int32_t num_candidates = 8400; // YOLOv8 native grid predictions
            constexpr int32_t num_classes = num_channels - 4;

            std::vector<Detection> candidates;
            candidates.reserve(256); // Sensible pre-allocation baseline for real-world scenarios

            // Native YOLOv8 FP32 ONNX outputs are strictly channels-first [1x84x8400]
            // We use direct linear probability indexing for massive performance gains
            for (int32_t col = 0; col < num_candidates; ++col)
            {
                float max_score = 0.0f;
                int32_t best_class_id = -1;

                // Gather class scores for this candidate grid cell column
                for (int32_t cls = 0; cls < num_classes; ++cls)
                {
                    const int32_t ch = cls + 4;
                    // Standard explicit channels_first stride mapping logic
                    size_t tensor_idx = (static_cast<size_t>(ch) * num_candidates) + col;
                    float score = raw_output[tensor_idx];

                    if (score > max_score)
                    {
                        max_score = score;
                        best_class_id = cls;
                    }
                }

                if (max_score < conf_threshold || best_class_id < 0)
                {
                    continue;
                }

                // Read continuous layout box geometry bounding fields safely
                float cx = raw_output[(0 * num_candidates) + col];
                float cy = raw_output[(1 * num_candidates) + col];
                float w = raw_output[(2 * num_candidates) + col];
                float h = raw_output[(3 * num_candidates) + col];

                if (w <= 0.0f || h <= 0.0f)
                {
                    continue;
                }

                float x1_canvas = cx - (w / 2.0f);
                float y1_canvas = cy - (h / 2.0f);
                float x2_canvas = cx + (w / 2.0f);
                float y2_canvas = cy + (h / 2.0f);

                Detection box;
                box.x1 = (x1_canvas - meta.pad_x) / meta.scale_factor;
                box.y1 = (y1_canvas - meta.pad_y) / meta.scale_factor;
                box.x2 = (x2_canvas - meta.pad_x) / meta.scale_factor;
                box.y2 = (y2_canvas - meta.pad_y) / meta.scale_factor;
                box.confidence = max_score;
                box.class_id = best_class_id;

                // Restrict variables natively safely within frame image borders
                box.x1 = std::clamp(box.x1, 0.0f, static_cast<float>(meta.src_width));
                box.y1 = std::clamp(box.y1, 0.0f, static_cast<float>(meta.src_height));
                box.x2 = std::clamp(box.x2, 0.0f, static_cast<float>(meta.src_width));
                box.y2 = std::clamp(box.y2, 0.0f, static_cast<float>(meta.src_height));

                if (box.x2 <= box.x1 || box.y2 <= box.y1)
                {
                    continue;
                }

                candidates.push_back(box);
            }

            if (candidates.empty())
            {
                return {};
            }

            // Step 2: Order candidate structures descendingly based on confidence values
            std::sort(candidates.begin(), candidates.end(), [](const Detection &a, const Detection &b)
                      { return a.confidence > b.confidence; });

            // Step 3: Fast Class-Specific Non-Maximum Suppression (NMS) Execution
            std::vector<Detection> final_detections;
            std::vector<uint8_t> is_suppressed(candidates.size(), 0);

            for (size_t i = 0; i < candidates.size(); ++i)
            {
                if (is_suppressed[i])
                {
                    continue;
                }

                final_detections.push_back(candidates[i]);

                for (size_t j = i + 1; j < candidates.size(); ++j)
                {
                    if (is_suppressed[j])
                    {
                        continue;
                    }

                    // Class-Specific Check: overlapping boxes of different classes won't cancel out
                    if (candidates[i].class_id == candidates[j].class_id)
                    {
                        if (ComputeIoU(candidates[i], candidates[j]) > iou_threshold)
                        {
                            is_suppressed[j] = 1;
                        }
                    }
                }
            }

            return final_detections;
        }
    }
}
