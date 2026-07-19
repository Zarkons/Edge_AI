#include "file_input_handler.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <cstring>

namespace fs = std::filesystem;

namespace obj_rec
{
    namespace app
    {
        namespace
        {
            std::string ResolveExistingRunfilePath(
                const bazel::tools::cpp::runfiles::Runfiles &runfiles,
                const std::string &token)
            {
                std::string abs_path = runfiles.Rlocation(token);
                if (!abs_path.empty() && fs::exists(abs_path))
                {
                    return abs_path;
                }

                // Under Bzlmod the main workspace can be materialized under "_main".
                std::string prefixed_token = "_main/" + token;
                abs_path = runfiles.Rlocation(prefixed_token);
                if (!abs_path.empty() && fs::exists(abs_path))
                {
                    return abs_path;
                }

                return "";
            }
        }

        FileInputHandler::FileInputHandler(char *argv0, const std::string &sample_token)
        {
            std::string error;
            m_runfiles.reset(bazel::tools::cpp::runfiles::Runfiles::Create(argv0, &error));
            if (!m_runfiles)
            {
                throw std::runtime_error("Failed to initialize Bazel Runfiles system: " + error);
            }

            DiscoverImages(sample_token);
        }

        bool FileInputHandler::IsImageFile(const std::string &filename) const
        {
            std::string lower = filename;
            std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
            return (lower.ends_with(".png") || lower.ends_with(".jpg") || lower.ends_with(".jpeg"));
        }

        void FileInputHandler::DiscoverImages(const std::string &sample_token)
        {
            std::string abs_sample_path = ResolveExistingRunfilePath(*m_runfiles, sample_token);
            if (abs_sample_path.empty())
            {
                throw std::runtime_error("Bazel runfiles could not find target token: " + sample_token);
            }

            fs::path calibration_dir = fs::path(abs_sample_path).parent_path();

            for (const auto &entry : fs::directory_iterator(calibration_dir))
            {
                if (m_image_paths.size() >= m_max_images)
                    break;

                std::string path_str = entry.path().string();
                if (IsImageFile(path_str))
                {
                    m_image_paths.push_back(path_str);
                }
            }

            if (m_image_paths.empty())
            {
                throw std::runtime_error("No valid image configurations found inside: " + calibration_dir.string());
            }

            m_current_image = m_image_paths.begin();
        }

        bool FileInputHandler::GetNext(RawBufferFrame &out_frame)
        {
            if (m_current_image == m_image_paths.end())
            {
                return false;
            }

            std::string path = *m_current_image;
            m_current_image++;

            // Read the image completely raw and unchanged from the disk file layout
            cv::Mat raw_img = cv::imread(path, cv::IMREAD_UNCHANGED);
            if (raw_img.empty())
            {
                std::cerr << "Warning: Skipping damaged asset path: " << path << std::endl;
                return GetNext(out_frame); // Tail recursion loop to secure the next valid frame
            }

            // Expose layout dimensions
            out_frame.width = raw_img.cols;
            out_frame.height = raw_img.rows;
            out_frame.channels = raw_img.channels();
            out_frame.stride = static_cast<int32_t>(raw_img.step);

            // Perform a safe contiguous byte copy into the pre-allocated vector payload
            size_t total_bytes = raw_img.total() * raw_img.elemSize();
            out_frame.data.resize(total_bytes);
            std::memcpy(out_frame.data.data(), raw_img.data, total_bytes);

            return true;
        }

        std::string FileInputHandler::GetModelPath(const std::string &model_token) const
        {
            std::string abs_model_path = ResolveExistingRunfilePath(*m_runfiles, model_token);
            if (abs_model_path.empty())
            {
                throw std::runtime_error("Bazel runfiles could not find target model token: " + model_token);
            }
            return abs_model_path;
        }
    }
}
