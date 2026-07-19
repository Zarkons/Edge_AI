#include "sample_processing_from_file.h"
#include "get_label_names.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

namespace
{
    namespace fs = std::filesystem;
    using namespace ml_types_dynamic;
    using namespace file_sample_processing;
    constexpr size_t kInputSampleCount = 16000;

    std::string getWavFilesPath()
    {
        const char *workspace_dir = std::getenv("BUILD_WORKSPACE_DIRECTORY");
        if (workspace_dir != nullptr)
        {
            return (fs::path(workspace_dir) / "demo_applications//simple_audio_recognition/train/build_dir/data/mini_speech_commands_extracted/mini_speech_commands").string();
        }

        return "demo_applications/simple_audio_recognition/train/build_dir/data/mini_speech_commands_extracted/mini_speech_commands";
    }

    std::vector<std::string> scanDirectory(const std::string &directory_path)
    {
        std::vector<std::string> wav_files;
        if (!fs::exists(directory_path) || !fs::is_directory(directory_path))
        {
            std::cerr << "Error: Directory does not exist: " << directory_path << std::endl;
            return wav_files;
        }

        for (const auto &entry : fs::recursive_directory_iterator(directory_path))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }

            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".wav")
            {
                wav_files.push_back(entry.path().string());
            }
        }

        return wav_files;
    }

    std::string pickRandomFile(const std::vector<std::string> &wav_files)
    {
        if (wav_files.empty())
        {
            return "";
        }

        static std::mt19937 rng(std::random_device{}());
        std::uniform_int_distribution<size_t> dist(0, wav_files.size() - 1);
        return wav_files[dist(rng)];
    }

    RawSignalFloatVec loadRawWav(const std::string &wav_path)
    {
        std::ifstream file(wav_path, std::ios::binary);
        if (!file.is_open())
        {
            return RawSignalFloatVec(kInputSampleCount, 0.0f);
        }

        // Skip the standard RIFF WAV header.
        file.seekg(44, std::ios::beg);

        RawSignalFloatVec raw_waveform;
        raw_waveform.reserve(kInputSampleCount);

        int16_t sample = 0;
        while (raw_waveform.size() < kInputSampleCount &&
               file.read(reinterpret_cast<char *>(&sample), sizeof(int16_t)))
        {
            raw_waveform.push_back(static_cast<float>(sample));
        }

        if (raw_waveform.size() < kInputSampleCount)
        {
            raw_waveform.resize(kInputSampleCount, 0.0f);
        }

        return raw_waveform;
    }
} // namespace

namespace file_sample_processing
{
    using namespace ml_types_dynamic;

    AudioSample GetSample()
    {
        const std::string directory_path = getWavFilesPath();
        std::vector<std::string> wav_files = scanDirectory(directory_path);

        AudioSample sample;
        sample.path = pickRandomFile(wav_files);
        if (sample.path.empty())
        {
            std::cerr << "Warning: No .wav files found in " << directory_path << std::endl;
            sample.raw_waveform = RawSignalFloatVec(kInputSampleCount, 0.0f);
            return sample;
        }

        sample.raw_waveform = loadRawWav(sample.path);
        return sample;
    }
}