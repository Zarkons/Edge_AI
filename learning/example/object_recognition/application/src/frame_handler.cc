#include "frame_handler.h"
#include "print_logger.h"
#include "TracyProfiler.h"
#

namespace obj_rec::app
{

    FrameHandler::~FrameHandler()
    {
        Stop();
    }

    bool FrameHandler::Initialize(const std::string &stream_url)
    {
        return camera_handler_.Initialize(stream_url);
    }

    void FrameHandler::Start()
    {
        if (is_running_.load())
            return;

        is_running_.store(true);
        worker_thread_ = std::thread(&FrameHandler::WorkerLoop, this);
        PRINT_INFO("[FrameHandler] Background ingestion thread started.");
    }

    void FrameHandler::Stop()
    {
        if (!is_running_.load())
            return;

        is_running_.store(false);
        if (worker_thread_.joinable())
        {
            worker_thread_.join();
        }
        PRINT_INFO("[FrameHandler] Background ingestion thread stopped.");
    }

    void FrameHandler::WorkerLoop()
    {
        CameraFrame local_frame;
        tracy::SetThreadName("FrameHandler Worker");

        while (is_running_.load())
        {
            ZoneScopedN("Worker Frame Fetch");

            if (!camera_handler_.GetNextFrame(local_frame))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            {
                LockGuardType(buffer_mutex_, "Worker Buffer Lock");
                std::swap(shared_buffer_, local_frame);
                has_new_frame_ = true;
            }
        }
    }

    bool FrameHandler::FetchLatestFrame(CameraFrame &out_frame)
    {
        ZoneScopedN("Consumer Frame Fetch");
        LockGuardType(buffer_mutex_, "Consumer Buffer Lock");
        if (!has_new_frame_)
        {
            return false;
        }

        std::swap(shared_buffer_, out_frame);
        has_new_frame_ = false;
        return true;
    }

} // namespace obj_rec::app
