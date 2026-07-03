#include "YOLOv8PostProcessor.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <array>

namespace
{
    enum class ScoreMode
    {
        kProbability,
        kSoftmax,
        kLogits
    };

    struct ScoreTransform
    {
        ScoreMode mode = ScoreMode::kProbability;
        float scale = 1.0f;
    };

    inline float Sigmoid(float x)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }

    inline float NormalizeScore(float score, const ScoreTransform &transform)
    {
        if (transform.mode == ScoreMode::kLogits)
        {
            score = Sigmoid(score);
        }

        return std::clamp(score, 0.0f, 1.0f);
    }
}

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
            float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
            float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);

            return intersection_area / (area_a + area_b - intersection_area);
        }

        std::vector<Detection> YOLOv8PostProcessor::Execute(const float *raw_output,
                                                            float conf_threshold,
                                                            float iou_threshold,
                                                            const dsp::image::TransformMetadata &meta)
        {
            std::vector<Detection> candidates;

            constexpr int32_t num_channels = 84;     // 4 box geometry dimensions + 80 COCO classes
            constexpr int32_t num_candidates = 8400; // YOLOv8 native grid predictions

            auto detect_score_mode = [&](bool channels_first_layout)
            {
                size_t outside_probability_range = 0;
                size_t inspected = 0;
                float min_score = std::numeric_limits<float>::max();
                float max_score = std::numeric_limits<float>::lowest();

                for (int32_t col = 0; col < num_candidates; ++col)
                {
                    for (int32_t ch = 4; ch < num_channels; ++ch)
                    {
                        size_t tensor_idx = channels_first_layout
                                                ? (static_cast<size_t>(ch) * num_candidates + col)
                                                : (static_cast<size_t>(col) * num_channels + ch);
                        float score = raw_output[tensor_idx];
                        min_score = std::min(min_score, score);
                        max_score = std::max(max_score, score);
                        if (score < 0.0f || score > 1.0f)
                        {
                            ++outside_probability_range;
                        }
                        ++inspected;
                    }
                }

                if (inspected == 0)
                {
                    return ScoreTransform{};
                }

                float out_ratio = static_cast<float>(outside_probability_range) / static_cast<float>(inspected);

                // Quantized exports may expose non-negative class logits/scores above 1.
                if (min_score >= 0.0f && max_score > 1.0f)
                {
                    return ScoreTransform{ScoreMode::kSoftmax, 1.0f};
                }

                // Logit tensors commonly contain substantial negative values.
                if (min_score < 0.0f && out_ratio > 0.20f)
                {
                    return ScoreTransform{ScoreMode::kLogits, 1.0f};
                }

                return ScoreTransform{};
            };

            auto decode_candidates = [&](bool channels_first_layout)
            {
                std::vector<Detection> decoded;
                decoded.reserve(static_cast<size_t>(num_candidates));
                const ScoreTransform score_transform = detect_score_mode(channels_first_layout);
                constexpr int32_t num_classes = num_channels - 4;
                std::array<float, num_classes> class_scores{};

                for (int32_t col = 0; col < num_candidates; ++col)
                {
                    float max_score = 0.0f;
                    int32_t best_class_id = -1;

                    // Gather class scores for this candidate.
                    for (int32_t cls = 0; cls < num_classes; ++cls)
                    {
                        const int32_t ch = cls + 4;
                        size_t tensor_idx = channels_first_layout
                                                ? (static_cast<size_t>(ch) * num_candidates + col)
                                                : (static_cast<size_t>(col) * num_channels + ch);
                        class_scores[static_cast<size_t>(cls)] = raw_output[tensor_idx];
                    }

                    if (score_transform.mode == ScoreMode::kSoftmax)
                    {
                        float max_logit = class_scores[0];
                        for (int32_t cls = 1; cls < num_classes; ++cls)
                        {
                            max_logit = std::max(max_logit, class_scores[static_cast<size_t>(cls)]);
                        }

                        float exp_sum = 0.0f;
                        for (int32_t cls = 0; cls < num_classes; ++cls)
                        {
                            float e = std::exp(class_scores[static_cast<size_t>(cls)] - max_logit);
                            class_scores[static_cast<size_t>(cls)] = e;
                            exp_sum += e;
                        }

                        if (exp_sum > 0.0f)
                        {
                            for (int32_t cls = 0; cls < num_classes; ++cls)
                            {
                                float score = class_scores[static_cast<size_t>(cls)] / exp_sum;
                                if (score > max_score)
                                {
                                    max_score = score;
                                    best_class_id = cls;
                                }
                            }
                        }
                    }
                    else
                    {
                        for (int32_t cls = 0; cls < num_classes; ++cls)
                        {
                            float score = NormalizeScore(class_scores[static_cast<size_t>(cls)], score_transform);
                            if (score > max_score)
                            {
                                max_score = score;
                                best_class_id = cls;
                            }
                        }
                    }

                    if (max_score < conf_threshold || best_class_id < 0)
                    {
                        continue;
                    }

                    // Read box geometry from the same layout.
                    float cx = channels_first_layout ? raw_output[0 * num_candidates + col] : raw_output[static_cast<size_t>(col) * num_channels + 0];
                    float cy = channels_first_layout ? raw_output[1 * num_candidates + col] : raw_output[static_cast<size_t>(col) * num_channels + 1];
                    float w = channels_first_layout ? raw_output[2 * num_candidates + col] : raw_output[static_cast<size_t>(col) * num_channels + 2];
                    float h = channels_first_layout ? raw_output[3 * num_candidates + col] : raw_output[static_cast<size_t>(col) * num_channels + 3];

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

                    box.x1 = std::clamp(box.x1, 0.0f, static_cast<float>(meta.src_width));
                    box.y1 = std::clamp(box.y1, 0.0f, static_cast<float>(meta.src_height));
                    box.x2 = std::clamp(box.x2, 0.0f, static_cast<float>(meta.src_width));
                    box.y2 = std::clamp(box.y2, 0.0f, static_cast<float>(meta.src_height));

                    // Reject degenerate boxes after clipping.
                    if (box.x2 <= box.x1 || box.y2 <= box.y1)
                    {
                        continue;
                    }

                    decoded.push_back(box);
                }

                return decoded;
            };

            // Prefer native YOLOv8 layout [1, 84, 8400] and only fallback when empty.
            candidates = decode_candidates(true);
            if (candidates.empty())
            {
                candidates = decode_candidates(false);
            }

            // 2. Descending Sort
            std::sort(candidates.begin(), candidates.end(), [](const Detection &a, const Detection &b)
                      { return a.confidence > b.confidence; });

            // 3. Execution of Non-Maximum Suppression (NMS)
            std::vector<Detection> final_detections;
            std::vector<bool> is_suppressed(candidates.size(), false);

            for (size_t i = 0; i < candidates.size(); ++i)
            {
                if (is_suppressed[i])
                    continue;

                final_detections.push_back(candidates[i]);

                for (size_t j = i + 1; j < candidates.size(); ++j)
                {
                    if (is_suppressed[j])
                        continue;

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
