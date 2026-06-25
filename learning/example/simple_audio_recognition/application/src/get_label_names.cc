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

    std::vector<std::string> getLabelNames()
    {
        const char *workspace_dir = std::getenv("BUILD_WORKSPACE_DIRECTORY");
        const std::string label_names_file = (workspace_dir != nullptr)
                                                 ? (fs::path(workspace_dir) / "learning/example/simple_audio_recognition/train/build_dir/label_names.txt").string()
                                                 : "learning/example/simple_audio_recognition/train/build_dir/label_names.txt";

        std::vector<std::string> labels;
        std::ifstream file(label_names_file);

        // If the file fails to open, log an error and return an empty vector to prevent crashing
        if (!file.is_open())
        {
            return labels;
        }

        std::string line;
        // Read the file line by line until EOF (End Of File)
        while (std::getline(file, line))
        {
            // Safety guard: skip empty lines if they exist in the text file
            if (!line.empty())
            {
                labels.push_back(line);
            }
        }

        return labels;
    }

} // namespace

namespace audio_sample_processing
{
    std::vector<std::string> GetLabelNamesList()
    {
        return getLabelNames();
    }
}