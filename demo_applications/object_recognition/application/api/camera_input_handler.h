#pragma once

#include <string>
#include <vector>
#include <deque>
#include <opencv2/opencv.hpp>

namespace obj_rec
{
    namespace app
    {
        struct CameraFrame
        {
            std::vector<uint8_t> data;
            int32_t width = 0;
            int32_t height = 0;
            int32_t channels = 3;
            int32_t stride = 0;
        };

        enum class CameraFrameStatus
        {
            kOk,           // Frame successfully decoded
            kRetry,        // Transient — no data yet, safe to retry
            kDisconnected, // Permanent — socket closed or fatal error
        };

        class CameraInputHandler
        {
        public:
            CameraInputHandler();
            ~CameraInputHandler();

            bool Initialize(const std::string &stream_url);
            CameraFrameStatus GetNextFrame(CameraFrame &out_frame);

        private:
            bool ParseUrl(const std::string &url, std::string &host, std::string &port, std::string &path);

            int m_sock_fd;
            std::deque<uint8_t> m_stream_buffer;
            std::vector<uint8_t> m_jpeg_cache;
        };
    }
}
