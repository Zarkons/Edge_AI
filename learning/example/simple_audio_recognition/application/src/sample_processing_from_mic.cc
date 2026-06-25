#include "sample_processing_from_mic.h"
#include <iostream>

namespace audio_sample_processing
{
    MicrophoneInput::MicrophoneInput()
    {
        audio_ring_buffer_.reserve(WINDOW_SIZE * 2);
    }

    MicrophoneInput::~MicrophoneInput()
    {
        Stop();
    }

    void MicrophoneInput::AudioCaptureCallback(ma_device *pDevice, void *pOutput, const void *pInput, ma_uint32 frameCount)
    {
        (void)pOutput;
        // static uint64_t timestamp = 0;
        // timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        //                 std::chrono::system_clock::now().time_since_epoch())
        //                 .count();
        // printf("AudioCaptureCallback called at timestamp: %llu ms\n", timestamp);
        // printf("Frame count: %u\n", frameCount);
        auto *instance = static_cast<MicrophoneInput *>(pDevice->pUserData);
        if (!instance || !pInput || frameCount == 0)
            return;

        const float *float_input = static_cast<const float *>(pInput);
        std::lock_guard<std::mutex> lock(instance->buffer_mutex_);

        for (ma_uint32 i = 0; i < frameCount; ++i)
        {
            instance->audio_ring_buffer_.push_back(float_input[i]);
        }

        if (instance->audio_ring_buffer_.size() > instance->WINDOW_SIZE)
        {
            size_t excess = instance->audio_ring_buffer_.size() - instance->WINDOW_SIZE;
            instance->audio_ring_buffer_.erase(
                instance->audio_ring_buffer_.begin(),
                instance->audio_ring_buffer_.begin() + excess);
        }
        instance->new_data_flag = true;
    }

    bool MicrophoneInput::Start()
    {
        ma_device_config config = ma_device_config_init(ma_device_type_capture);
        config.capture.format = ma_format_f32;
        config.capture.channels = 1; // Request exactly 1 channel (Mono)
        config.sampleRate = 16000;   // 16kHz audio sample rate
        config.dataCallback = AudioCaptureCallback;
        config.pUserData = this;

        // 🔥 CORRECTED FIXED HARDWARE COMPATIBILITY CODE:
        // Forces macOS CoreAudio to share the resource and cleanly mix down
        // any physical hardware stereo/multi-mic arrays down to a single channel.
        config.capture.shareMode = ma_share_mode_shared;
        config.noClip = MA_TRUE;

        // (The broken ma_channel_map_init_standard and memcpy lines have been removed completely!)

        if (ma_device_init(NULL, &config, &device_) != MA_SUCCESS)
        {
            std::cerr << "Error: Failed to open default mic hardware device." << std::endl;
            return false;
        }

        return ma_device_start(&device_) == MA_SUCCESS;
    }

    void MicrophoneInput::Stop()
    {
        if (is_initialized_)
        {
            ma_device_uninit(&device_);
            is_initialized_ = false;
        }
    }

    bool MicrophoneInput::HasNewData()
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        return new_data_flag;
    }

    std::vector<float> MicrophoneInput::GetLatestWindow()
    {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        if (audio_ring_buffer_.size() < WINDOW_SIZE)
        {
            return std::vector<float>(WINDOW_SIZE, 0.0f);
        }
        new_data_flag = false;
        return audio_ring_buffer_;
    }
}