#include "audio_recognition_model.h"
#include "tflite_model_handler.h"
#include "sample_processing_from_file.h"
#include "audio_model_pipeline.h"
#include "common_types.h"
#include "get_label_names.h"

#define TENSOR_ARENA_SIZE 512 * 1024

using namespace audio_model_pipeline;
using namespace quantization_types;
using namespace file_sample_processing;
using namespace tflite_model_handler;

int main()
{
    printf("Starting simple audio recognition example...\n");
    bool result = false;
    FeatureFloatVec spectrogram;
    PredictionFloatVec predictions;

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

    uint32_t correct_predictions = 0;
    uint32_t total_predictions = 0;
    for (int i = 0; i < 100; ++i)
    {

        AudioSample sample = GetSample();
        // printf("Random WAV file: %s\n", sample.path.c_str());
        // printf("Raw waveform size: %zu\n", sample.raw_waveform.size());
        spectrogram = ComputeSTFT(sample.raw_waveform);
        // printf("Spectrogram size: %zu\n", spectrogram.size());

        float spectogram_gain = 1.0f; // Adjust this value to boost or reduce the spectrogram values
        predictions = RunInference(audio_model, spectrogram, spectogram_gain);
        // printf("Predictions size: %zu\n", predictions.size());

        auto max_iterator = std::max_element(predictions.begin(), predictions.end());

        size_t winning_index = std::distance(predictions.begin(), max_iterator);
        // float highest_confidence = *max_iterator;
        std::vector<std::string> label_names = GetLabelNamesList();
        // printf("Predicted label: %s, confidence: %.4f\n", label_names[winning_index].c_str(), highest_confidence);
        if (sample.path.find(label_names[winning_index]) != std::string::npos)
        {
            // printf("Prediction is correct!\n");
            ++correct_predictions;
        }
        else
        {
            // printf("Prediction is incorrect.\n");
        }
        ++total_predictions;
        printf("Progress: %u/%u predictions made.\n", total_predictions, 100);
    }
    printf("Final accuracy: %.2f%% (%u/%u correct predictions)\n", (static_cast<float>(correct_predictions) / total_predictions) * 100.0f, correct_predictions, total_predictions);
}