#ifndef SAMPLE_PROCESSING_FROM_FILE_H_
#define SAMPLE_PROCESSING_FROM_FILE_H_

#include <string>
#include <vector>
#include "common_types.h"

/**
 * @namespace file_sample_processing
 * @brief Provides utilities for loading, managing, and parsing raw audio sample files.
 *
 * This namespace isolates the file I/O operations from downstream neural network pipeline stages,
 * transforming structural file metadata and raw data buffers into standardized ML types.
 */
namespace file_sample_processing
{
    /**
     * @struct AudioSample
     * @brief Container representing an individual isolated audio track and its structural metadata.
     */
    struct AudioSample
    {
        /** @brief Absolute or relative path to the physical audio file (typically a .wav format). */
        std::string path;

        /** @brief Normalized 1D linear amplitude vector of the loaded waveform data. */
        ml_types_dynamic::RawSignalFloatVec raw_waveform;
    };

    /**
     * @brief Fetches and processes an audio sample from the underlying processing pipeline.
     *
     * This function encapsulates sample discovery or queue popping logic. It handles disk read operations,
     * performs channel coercion, downsamples the binary structure to matching system specs, and packs the
     * results into an operational struct.
     *
     * @return AudioSample A populated struct containing the audio path and its floating-point data.
     * @throw std::runtime_error If the file stream is corrupted, missing, or fails structural parsing.
     */
    AudioSample GetSample();

} // namespace file_sample_processing

#endif // SAMPLE_PROCESSING_FROM_FILE_H_