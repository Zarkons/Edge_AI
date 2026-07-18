#ifndef OBJ_REC_APP_FILE_INPUT_HANDLER_H_
#define OBJ_REC_APP_FILE_INPUT_HANDLER_H_

#include <string>
#include <vector>
#include <memory>
#include "tools/cpp/runfiles/runfiles.h"

namespace obj_rec
{
    namespace app
    {
        /**
         * @struct RawBufferFrame
         * @brief Container wrapping a self-managed vector byte buffer.
         */
        struct RawBufferFrame
        {
            std::vector<uint8_t> data; /**< The managed contiguous byte array holding pixels. */
            int32_t width = 0;
            int32_t height = 0;
            int32_t channels = 0;
            int32_t stride = 0;
        };

        class FileInputHandler
        {
        public:
            FileInputHandler(char *argv0, const std::string &sample_token);
            ~FileInputHandler() = default;

            /**
             * @brief Fetches the next disk asset and copies its decoded pixels into the vector.
             * @param[out] out_frame Target container where properties and bytes will be written.
             * @return false when the end of the directory stream is reached.
             */
            bool GetNext(RawBufferFrame &out_frame);

            size_t GetTotalImagesCount() const { return m_image_paths.size(); }

            /**
             * @brief Resolves the absolute system path for a model from its Bazel token.
             * @param model_token The runfiles lookup key.
             * @return Absolute path to the model file on disk.
             */
            std::string GetModelPath(const std::string &model_token) const;

        private:
            bool IsImageFile(const std::string &filename) const;
            void DiscoverImages(const std::string &sample_token);

            std::unique_ptr<bazel::tools::cpp::runfiles::Runfiles> m_runfiles;
            std::vector<std::string> m_image_paths;
            std::vector<std::string>::const_iterator m_current_image;
            const size_t m_max_images = 100;
        };
    }
}

#endif // OBJ_REC_APP_FILE_INPUT_HANDLER_H_
