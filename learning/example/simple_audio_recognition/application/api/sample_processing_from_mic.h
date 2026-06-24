#ifndef SAMPLE_PROCESSING_FROM_MIC_H_
#define SAMPLE_PROCESSING_FROM_MIC_H_

#include "miniaudio.h"
#include <vector>
#include <mutex>

namespace AudioSampleProcessing
{

    class MicrophoneInput
    {
    private:
        ma_device device_;
        bool is_initialized_ = false;
        bool new_data_flag = false;

        std::vector<float> audio_ring_buffer_;
        std::mutex buffer_mutex_;

        const size_t WINDOW_SIZE = 16000;

        static void AudioCaptureCallback(ma_device *pDevice, void *pOutput, const void *pInput, ma_uint32 frameCount);

    public:
        MicrophoneInput();
        ~MicrophoneInput();

        bool Start();
        void Stop();
        std::vector<float> GetLatestWindow();
        bool HasNewData();
    };
}
#endif // SAMPLE_PROCESSING_FROM_MIC_H_
