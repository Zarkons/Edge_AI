#include "yolov8_visualizer.h"

namespace obj_rec
{
    namespace app
    {
        void YOLOv8Visualizer::DrawDetections(cv::Mat &image, const std::vector<Detection> &detections, const std::vector<std::string> &class_names)
        {
            for (const auto &box : detections)
            {
                int32_t offset = box.class_id * 12345;
                cv::Scalar color(
                    (offset & 0xFF),
                    ((offset >> 8) & 0xFF),
                    ((offset >> 16) & 0xFF));

                int x = static_cast<int>(box.x1);
                int y = static_cast<int>(box.y1);
                int w = static_cast<int>(box.x2 - box.x1);
                int h = static_cast<int>(box.y2 - box.y1);
                cv::Rect rect(x, y, w, h);

                cv::rectangle(image, rect, color, 2, cv::LINE_AA);

                // FIX: Map Class ID to its text name if it falls within array limits
                std::string class_label;
                if (box.class_id >= 0 && static_cast<size_t>(box.class_id) < class_names.size())
                {
                    class_label = class_names[box.class_id];
                }
                else
                {
                    class_label = "Class " + std::to_string(box.class_id);
                }

                std::string label = class_label + ": " + cv::format("%.1f", box.confidence * 100.0f) + "%";

                int font_face = cv::FONT_HERSHEY_SIMPLEX;
                double font_scale = 0.5;
                int thickness = 1;
                int baseline = 0;
                cv::Size text_size = cv::getTextSize(label, font_face, font_scale, thickness, &baseline);

                int label_y = std::max(y, text_size.height + 5);
                cv::Rect text_bg(x, label_y - text_size.height - 5, text_size.width + 6, text_size.height + 6);

                cv::rectangle(image, text_bg, color, cv::FILLED);
                cv::putText(image, label, cv::Point(x + 3, label_y - 2),
                            font_face, font_scale, cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
            }
        }
    }
}