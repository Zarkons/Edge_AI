#ifndef ML_MODEL_RESOLVER_H_
#define ML_MODEL_RESOLVER_H_

#include <string>

namespace obj_rec
{
    namespace app
    {
        /**
         * @brief Resolves the deployment model target to an absolute path on disk.
         * @param argv0 The application's entry point path pointer (argv[0]).
         * @return The absolute filesystem path string to the target model file.
         */
        [[nodiscard]] std::string ResolveAbsModelPath(char *argv0);
    } // namespace app
} // namespace obj_rec

#endif // ML_MODEL_RESOLVER_H_
