#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "dip_data_types.h"

namespace obj_rec
{
    namespace app
    {
        /**
         * @struct CameraFrame
         * @brief Container wrapping a self-managed vector byte buffer for live camera frames.
         */
        struct CameraFrame
        {
            std::vector<uint8_t> data;
            int32_t width = 0;
            int32_t height = 0;
            int32_t channels = 3;
            int32_t stride = 0;
        };

        class CameraInputHandler
        {
        public:
            CameraInputHandler() = default;
            ~CameraInputHandler();

            /**
             * @brief Connects to the iPhone IP Camera stream.
             * @param stream_url The full network URL (e.g., "rtsp://111.111.1.11:8080/h264" or "http://111.111.1")
             * @return true if connection is established successfully, false otherwise.
             */
            bool Initialize(const std::string &stream_url);

            /**
             * @brief Captures the latest live network frame into the target struct.
             * @param out_frame Struct where raw BGR24 frame metadata and pixels will be allocated.
             * @return true if a frame was successfully captured, false if the stream disconnected.
             */
            bool GetNextFrame(CameraFrame &out_frame);

        private:
            cv::VideoCapture m_cap;
        };
    }
}