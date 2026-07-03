#include "CameraInputHandler.h"
#include <iostream>

namespace obj_rec
{
    namespace app
    {
        CameraInputHandler::~CameraInputHandler()
        {
            if (m_cap.isOpened())
            {
                m_cap.release();
            }
        }

        bool CameraInputHandler::Initialize(const std::string &stream_url)
        {
            std::cout << "[CameraInputHandler] Connecting to network stream: " << stream_url << " ..." << std::endl;

            // Open the network capture pipeline
            m_cap.open(stream_url, cv::CAP_ANY);

            if (!m_cap.isOpened())
            {
                std::cerr << "[CameraInputHandler Error] Failed to open stream at " << stream_url
                          << ". Double check your iPhone's IP Address and network connection." << std::endl;
                return false;
            }

            // Optimize latency for real-time live streaming applications
            m_cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

            std::cout << "[CameraInputHandler] Connected successfully! Fetching properties..." << std::endl;
            std::cout << "   - Target Source Dimensions: "
                      << m_cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
                      << m_cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;

            return true;
        }

        bool CameraInputHandler::GetNextFrame(CameraFrame &out_frame)
        {
            cv::Mat raw_mat;

            // Grab and decode the latest frame from the live network pipeline
            if (!m_cap.read(raw_mat) || raw_mat.empty())
            {
                return false;
            }

            // Enforce 3-channel continuous BGR configurations
            if (raw_mat.type() != CV_8UC3)
            {
                cv::cvtColor(raw_mat, raw_mat, cv::COLOR_BGRA2BGR);
            }

            out_frame.width = raw_mat.cols;
            out_frame.height = raw_mat.rows;
            out_frame.channels = 3;
            out_frame.stride = static_cast<int32_t>(raw_mat.step);

            // Resize the flat memory vector to fit the matrix requirements
            size_t total_bytes = out_frame.stride * out_frame.height;
            out_frame.data.resize(total_bytes);

            // Copy raw pixel array out of the continuous OpenCV matrix container
            std::memcpy(out_frame.data.data(), raw_mat.data, total_bytes);

            return true;
        }
    }
}