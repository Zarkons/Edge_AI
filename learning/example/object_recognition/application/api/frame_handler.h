#ifndef FRAME_HANDLER_H_
#define FRAME_HANDLER_H_

#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <condition_variable>
#include "camera_input_handler.h"
#include "dip_data_types.h"
#include "TracyProfiler.h"

namespace obj_rec::app
{

    class FrameHandler
    {
    public:
        FrameHandler() = default;
        ~FrameHandler();

        // Prevent copying for thread safety
        FrameHandler(const FrameHandler &) = delete;
        FrameHandler &operator=(const FrameHandler &) = delete;

        // Lifecyle API
        bool Initialize(const std::string &stream_url);
        void Start();
        void Stop();
        bool IsRunning() const { return is_running_.load(); }

        // Consumer API: Non-blocking check for new data
        bool FetchLatestFrame(CameraFrame &out_frame);

    private:
        void WorkerLoop();

        CameraInputHandler camera_handler_;
        std::thread worker_thread_;
        std::atomic<bool> is_running_{false};

        // Triple buffering primitives to maximize throughput
        std::mutex buffer_mutex_;
        CameraFrame shared_buffer_;
        bool has_new_frame_{false};
    };

} // namespace obj_rec::app

#endif // FRAME_HANDLER_H_
