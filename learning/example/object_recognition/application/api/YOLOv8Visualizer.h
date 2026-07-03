#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "YOLOv8PostProcessor.h"

namespace obj_rec
{
    namespace app
    {
        class YOLOv8Visualizer
        {
        public:
            // Delete constructors to prevent instantiation since this is a static utility class
            YOLOv8Visualizer() = delete;
            ~YOLOv8Visualizer() = delete;

            /**
             * @brief Draws bounding boxes, class labels, and confidences directly onto an OpenCV Mat image.
             * @param image The input/output BGR image matrix to render onto.
             * @param detections A vector containing processed and filtered detections.
             * @param class_names A vector of class names corresponding to the detection class IDs.
             */
            static void DrawDetections(cv::Mat &image, const std::vector<Detection> &detections, const std::vector<std::string> &class_names = {});
        };
    }
}