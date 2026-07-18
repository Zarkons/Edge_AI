#include "ml_model_resolver.h"
#include <iostream>
#include <memory>
#include "tools/cpp/runfiles/runfiles.h"

namespace obj_rec::app
{
    std::string ResolveAbsModelPath(char *argv0)
    {
        const std::string kModelToken = "_main/learning/example/object_recognition/quantization/_build_dir/models/yolov8n_fp32.onnx";

        std::string runfiles_error;
        // Instantiate the Bazel Runfiles lookup library
        std::unique_ptr<bazel::tools::cpp::runfiles::Runfiles> runfiles(
            bazel::tools::cpp::runfiles::Runfiles::Create(argv0, &runfiles_error));

        if (!runfiles)
        {
            std::cerr << "[PATH RESOLVER] Failed to initialize Bazel Runfiles helper context: "
                      << runfiles_error << std::endl;
            return "";
        }

        // Search the runtime symlink tree for the absolute path of our token
        std::string abs_path = runfiles->Rlocation(kModelToken);
        if (abs_path.empty())
        {
            std::cerr << "[PATH RESOLVER] CRITICAL ERROR: Could not resolve physical filesystem location for token: "
                      << kModelToken << std::endl;
            return "";
        }

        std::cout << "[PATH RESOLVER] Deployment asset located successfully at: " << abs_path << std::endl;
        return abs_path;
    }
} // namespace obj_rec::app
