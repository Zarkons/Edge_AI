#ifndef AUDIO_MODEL_PIPELINE_H_
#define AUDIO_MODEL_PIPELINE_H_

#include <vector>
#include <cstdint>
#include "tflite_model_handler.h"
#include "common_types.h"

/**
 * @namespace audio_model_pipeline
 * @brief Handles 1D PCM audio feature extraction and INT8 quantized neural network inference pipelines.
 */
namespace audio_model_pipeline
{
    /** @brief Required length of the input audio buffer (1.0 second at 16kHz). */
    constexpr size_t INPUT_SAMPLE_COUNT = 16000;

    /** @brief Total time-slice frames generated across the sliding temporal layout. */
    constexpr size_t SPECTROGRAM_ROWS = 124;

    /** @brief Discrete frequency channels computed per frame slice. */
    constexpr size_t SPECTROGRAM_COLS = 129;

    /** @brief Total length of the flattened 1D feature array (124 rows * 129 columns = 15,996 elements). */
    constexpr size_t TOTAL_INPUT_ELEMENTS = SPECTROGRAM_ROWS * SPECTROGRAM_COLS;

    /**
     * @brief Pre-computes the static Hann Window coefficients matching TensorFlow's default signal layout.
     *
     * @warning This function must be executed exactly once at application startup before invoking `ComputeSTFT()`.
     *          Failure to initialize will result in processing empty/zero-volume window frames.
     */
    void HannWindowInit(void);

    /**
     * @brief Transforms a raw 1D audio waveform into a flattened 2D magnitude spectrogram vector.
     *
     * Normalizes the source PCM values from 16-bit range down to `[-1.0f, 1.0f]` in-place. The algorithm
     * slides a 256-sample window with a 127-sample step (~50% overlap), applies the pre-computed Hann coefficients,
     * and executes a Discrete Fourier Transform (DFT) to isolate spectral magnitudes.
     *
     * @param raw_waveform The source time-domain signal array. Modified in-place during scale normalization.
     * @return ml_types_dynamic::FeatureFloatVec Flattened feature magnitude vector of size exactly `TOTAL_INPUT_ELEMENTS` (15,996).
     * @note This function uses a thread-safe local buffer cache for concurrent execution support.
     */
    ml_types_dynamic::FeatureFloatVec ComputeSTFT(ml_types_dynamic::RawSignalFloatVec &raw_waveform);

    /**
     * @brief Calibrates, scales, and quantizes a float spectrogram into an INT8 matrix to execute model inference.
     *
     * Validates input sizing requirements, boosts weak signals by applying `spectrogram_gain`, maps values
     * into target integer limits using the runtime engine's quantization scale and zero-point parameters, and enforces strict
     * clamping within signed INT8 boundaries `[-128, 127]`.
     *
     * @param engine Active pointer to the generic multi-threaded TFLite runtime execution handler.
     * @param float_spectrogram Pre-calculated 1D float feature magnitude vector. Must match `TOTAL_INPUT_ELEMENTS`.
     * @param spectrogram_gain Scaling multiplier applied to pull small values out of the rounding dead-zone.
     * @return ml_types_dynamic::PredictionFloatVec Final network output classification probabilities, or an empty vector if sizing verification fails.
     */
    ml_types_dynamic::PredictionFloatVec RunInference(
        tflite_model_handler::TFLiteModelHandler<10> *engine,
        const ml_types_dynamic::FeatureFloatVec &float_spectrogram,
        const float spectrogram_gain);
} // namespace audio_model_pipeline

#endif // AUDIO_MODEL_PIPELINE_H_
