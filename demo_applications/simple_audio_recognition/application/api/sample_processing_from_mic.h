#ifndef SAMPLE_PROCESSING_FROM_MIC_H_
#define SAMPLE_PROCESSING_FROM_MIC_H_

#include "miniaudio.h"
#include <vector>
#include <mutex>

/**
 * @namespace mic_sample_processing
 * @brief Handles real-time hardware audio capture pipelines.
 */
namespace mic_sample_processing
{
    /**
     * @class MicrophoneInput
     * @brief Manages real-time 16kHz Mono audio streaming inputs utilizing miniaudio.
     *
     * This class implements an asynchronous producer-consumer architecture. A high-priority
     * hardware audio callback appends raw floating-point samples to an internal circular tracking
     * array, which is read safely on user-space worker threads to feed neural network pipelines.
     *
     * @note All internal state updates, metadata structures, and hardware configuration properties
     *       are fully thread-safe.
     */
    class MicrophoneInput
    {
    private:
        /** @brief Driver device container reference handle. */
        ma_device device_;

        /** @brief Thread-safe flag monitoring if miniaudio context initialization has succeeded. */
        bool is_initialized_ = false;

        /** @brief Data readiness sentinel flag signaled by the incoming hardware callback thread. */
        bool new_data_flag = false;

        /** @brief Linear memory heap tracking raw streaming mono audio amplitude patterns. */
        std::vector<float> audio_ring_buffer_;

        /** @brief Data mutex synchronization boundary safeguarding cross-thread array mutations. */
        std::mutex buffer_mutex_;

        /** @brief Constant target network envelope requirement constraint (16000 samples = 1.0 second). */
        const size_t WINDOW_SIZE = 16000;

        /**
         * @brief Standard miniaudio real-time audio sample interruption handling routine.
         *
         * Runs context switches automatically on dedicated kernel-level processing priorities.
         * Appends raw multi-channel mixed PCM buffers to user-space class structures.
         *
         * @warning This callback must perform minimal vector reallocations to avoid stalling audio registers.
         */
        static void AudioCaptureCallback(ma_device *pDevice, void *pOutput, const void *pInput, ma_uint32 frameCount);

    public:
        /**
         * @brief Instantiate the controller and pre-allocate internal ring buffer capacities.
         */
        MicrophoneInput();

        /**
         * @brief Joins trailing device threads and releases physical audio hardware handles.
         */
        ~MicrophoneInput();

        /**
         * @brief Configures shared-mode CoreAudio/ALSA hardware maps and enables streaming inputs.
         * @return true if the physical hardware device successfully initializes and boots up.
         * @return false if driver binding fails or exclusive access layers block input pipelines.
         */
        bool Start();

        /**
         * @brief Shuts down active data processing operations and closes down the system device structure.
         */
        void Stop();

        /**
         * @brief Checks if fresh, unread audio snapshots have arrived since the last state clearance.
         * @return true if new floating-point frames are ready for downstream classification.
         */
        bool HasNewData();

        /**
         * @brief Extracts a thread-safe snapshot vector copy matching the precise `WINDOW_SIZE` specification.
         *
         * Resets the underlying internal state flag `new_data_flag` to false upon read completion.
         * If the ring buffer contains fewer samples than required, it fills the output array with silence.
         *
         * @return std::vector<float> An array copy tracking the most recent 1-second chronological timeline.
         */
        std::vector<float> GetLatestWindow();
    };
} // namespace mic_sample_processing
#endif // SAMPLE_PROCESSING_FROM_MIC_H_
