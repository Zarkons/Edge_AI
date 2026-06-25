#include "audio_recognition_model.h"
#include "tflite_model_handler.h"
#include "sample_processing_from_mic.h"
#include "audio_model_pipeline.h"
#include "common_types.h"
#include "get_label_names.h"
#include <thread>
#include <numeric>

#define TENSOR_ARENA_SIZE 512 * 1024

using namespace audio_model_pipeline;
using namespace quantization_types;
using namespace audio_sample_processing;
using namespace tflite_model_handler;

int main()
{
    printf("Starting simple audio recognition example Mic input...\n");
    bool result = false;
    FeatureFloatVec spectrogram;
    PredictionFloatVec predictions;
    std::vector<std::string> label_names = GetLabelNamesList();

    TFLiteModelHandler<10> *audio_model = CreateTFLiteModelHandler<TENSOR_ARENA_SIZE, 10>();
    audio_model->RegisterOp([](TFLiteModelHandler<10>::ResolverType &resolver)
                            {
        resolver.AddConv2D();
        resolver.AddMaxPool2D();
        resolver.AddReshape();
        resolver.AddFullyConnected();
        resolver.AddStridedSlice();
        resolver.AddResizeBilinear();
        resolver.AddShape();
        resolver.AddPack(); });
    result = audio_model->Init(learning_example_simple_audio_recognition_application_tflite_model_audio_recognition_model_tflite);
    printf("Model initialization result: %s\n", result ? "Success" : "Failure");
    HannWindowInit();

    MicrophoneInput mic;
    if (!mic.Start())
    {
        printf("Error: Failed to start microphone input.\n");
        return -1;
    }

    printf("Listening for audio input from the microphone, speak now...\n");

    constexpr float CONFIDENCE_THRESHOLD = 0.7f;
    constexpr float NOISE_GATE_THRESHOLD = 0.007f;
    constexpr float SIGNAL_GAIN = 2.0f;
    constexpr float SPECTROGRAM_GAIN = 35000.0f;

    while (true)
    {
        // ====================================================================
        // THE HARDWARE SYNC GATE: Yield thread execution until miniaudio populates new data
        // ====================================================================
        if (!mic.HasNewData())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue; // Go back and wait, zero CPU processing overhead
        }

        // Fetch our perfectly bounded 16,000 element sliding window
        std::vector<float> current_audio = mic.GetLatestWindow();

        // Ensure the buffer is fully primed before starting inference
        if (current_audio.size() < 16000)
        {
            continue;
        }

        // 1. Apply raw signal gain to the sliding waveform snapshot
        for (float &sample : current_audio)
        {
            sample *= SIGNAL_GAIN;
        }

        // float current_max = *std::max_element(current_audio.begin(), current_audio.end());
        // float current_min = *std::min_element(current_audio.begin(), current_audio.end());

        // 2. Calculate average volume (Root Mean Square / Energy)
        float energy_sum = 0.0f;
        for (float sample : current_audio)
        {
            energy_sum += sample * sample;
        }
        float average_volume = std::sqrt(energy_sum / current_audio.size());

        // 3. Noise Gate Evaluation
        if (average_volume < NOISE_GATE_THRESHOLD)
        {
            // Keep console tight; skip printing dividers during ambient silence
            continue;
        }

        // printf("-----------------------------------------\n");
        // printf("Raw Sample Max: %f, Min: %f\n", current_max, current_min);

        // 4. Feature Extraction & Inference Pass
        std::vector<float> features = audio_model_pipeline::ComputeSTFT(current_audio);
        std::vector<float> probabilities = audio_model_pipeline::RunInference(audio_model, features, SPECTROGRAM_GAIN);

        if (!probabilities.empty())
        {
            auto max_iterator = std::max_element(probabilities.begin(), probabilities.end());
            size_t winning_index = std::distance(probabilities.begin(), max_iterator);
            float confidence = *max_iterator;
            std::string detected_word = label_names[winning_index];

            // 1. Setup a persistent tracking window across loop cycles
            static std::string best_candidate_word = "";
            static float best_candidate_confidence = 0.0f;
            static int frame_counter = 0;

            // Only track frames where the network actually detects a clean keyword shape
            if (confidence >= 0.25f)
            {
                if (confidence > best_candidate_confidence)
                {
                    best_candidate_confidence = confidence;
                    best_candidate_word = detected_word;
                }
                frame_counter++;
            }
            else
            {
                // If it's a silent or garbage frame, slowly decay the tracking memory
                frame_counter = std::max(0, frame_counter - 1);
            }

            // 2. Every 3 consecutive voice frames, evaluate the true acoustic peak
            if (frame_counter >= 3)
            {
                if (best_candidate_confidence >= CONFIDENCE_THRESHOLD)
                {
                    printf("-----------------------------------------\n");
                    printf(">>> SMOOTH DECISION -> Word: %s, Peak Confidence: %.4f\n",
                           best_candidate_word.c_str(), best_candidate_confidence);

                    if (best_candidate_word == "stop")
                    {
                        printf("Detected 'stop' command, exiting program...\n");
                        break;
                    }

                    // Flush the pipeline sequence by sleeping through the vowel tail
                    std::this_thread::sleep_for(std::chrono::milliseconds(300));
                }
                else
                {
                    printf("-----------------------------------------\n");
                    printf("Top word was %s (%.4f), but fell short of threshold.\n",
                           best_candidate_word.c_str(), best_candidate_confidence);
                }

                // Reset trackers for the next rolling window sequence
                best_candidate_word = "";
                best_candidate_confidence = 0.0f;
                frame_counter = 0;
            }
        }
    }

    printf("Exiting program, cleaning up resources...\n");
    mic.Stop();
    delete audio_model;
    return 0;
}