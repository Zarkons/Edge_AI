#include "audio_model_pipeline.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace audio_model_pipeline
{
    using namespace quantization_types;
    using namespace ml_types_dynamic;
    using namespace tflite_model_handler;

    constexpr size_t FRAME_LENGTH = 256;
    constexpr size_t FRAME_STEP = 127;

    static std::vector<float> hann_window(FRAME_LENGTH);
    thread_local static std::vector<float> windowed_signal(FRAME_LENGTH, 0.0f);

    void HannWindowInit(void)
    {
        // Pre-compute Hann Window coefficients (TensorFlow default)
        for (size_t i = 0; i < FRAME_LENGTH; ++i)
        {
            hann_window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (FRAME_LENGTH - 1)));
        }
    }

    FeatureFloatVec ComputeSTFT(RawSignalFloatVec &raw_waveform)
    {
        // 1. Normalize the raw waveform to [-1, 1] range
        for (float &sample : raw_waveform)
        {
            sample /= 32768.0f;
        }

        FeatureFloatVec spectrogram;
        spectrogram.reserve(TOTAL_INPUT_ELEMENTS);

        // 2. Compute window frames sequentially
        for (size_t frame = 0; frame < SPECTROGRAM_ROWS; ++frame)
        {
            size_t start_idx = frame * FRAME_STEP;

            // Apply windowing slice function
            for (size_t i = 0; i < FRAME_LENGTH; ++i)
            {
                size_t current_idx = start_idx + i;
                if (current_idx < raw_waveform.size())
                {
                    windowed_signal[i] = raw_waveform[current_idx] * hann_window[i];
                }
            }

            // 3. Compute Discrete Fourier Transform for the 129 non-redundant frequency channels
            for (size_t k = 0; k < SPECTROGRAM_COLS; ++k)
            {
                float real_sum = 0.0f;
                float imag_sum = 0.0f;

                for (size_t n = 0; n < FRAME_LENGTH; ++n)
                {
                    float angle = (2.0f * M_PI * k * n) / FRAME_LENGTH;
                    real_sum += windowed_signal[n] * std::cos(angle);
                    imag_sum -= windowed_signal[n] * std::sin(angle);
                }

                // Calculate magnitude matching TF tensor shape expectations
                float magnitude = std::sqrt(real_sum * real_sum + imag_sum * imag_sum);
                spectrogram.push_back(magnitude);
            }
        }

        return spectrogram;
    }

    PredictionFloatVec RunInference(
        TFLiteModelHandler<10> *engine,
        const FeatureFloatVec &float_spectrogram, const float spectrogram_gain)
    {
        QuantizationParams input_quant_params = engine->GetInputQuantizationParams();
        // Ensure data buffer sizing safety bounds check
        if (float_spectrogram.size() != TOTAL_INPUT_ELEMENTS)
        {
            return {};
        }

        // Allocate processing space for quantized values
        FeatureQuantInt8Vec quantized_buffer(TOTAL_INPUT_ELEMENTS);

        // Run INT8 mapping arithmetic
        for (size_t i = 0; i < TOTAL_INPUT_ELEMENTS; ++i)
        {
            // Apply the boost to pull the features out of the rounding dead-zone
            float boosted_feature = float_spectrogram[i] * spectrogram_gain;

            float scaled_val = boosted_feature / input_quant_params.scale;
            QuantizedInt8Type quantized_val = static_cast<QuantizedInt8Type>(std::round(scaled_val)) + input_quant_params.zero_point;

            // Force wrap limits into valid int8 bounds range [-128, 127]
            quantized_val = std::max(static_cast<QuantizedInt8Type>(-128), std::min(static_cast<QuantizedInt8Type>(127), quantized_val));
            quantized_buffer[i] = quantized_val;
        }

        // int zero_count = 0;
        // int positive_count = 0;
        // int negative_count = 0;
        // for (int8_t val : quantized_buffer)
        // {
        //     if (val == input_quant_params.zero_point)
        //         zero_count++;
        //     else if (val > input_quant_params.zero_point)
        //         positive_count++;
        //     else
        //         negative_count++;
        // }
        // printf("QUANT DICTIONARY -> Zero-Point Fill: %d, Highs: %d, Lows: %d\n", zero_count, positive_count, negative_count);

        // Call the data-agnostic engine using the quantized buffer and return the final float predictions
        return engine->Predict(quantized_buffer.data(), quantized_buffer.size() * sizeof(QuantizedInt8Type), true);
    }

} // namespace audio_model_pipeline
