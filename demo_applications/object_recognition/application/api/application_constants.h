#ifndef APPLICATION_CONSTANTS_H_
#define APPLICATION_CONSTANTS_H_

#include <cstdint>
#include <cstddef>

namespace obj_rec
{
    namespace app
    {
        inline constexpr int32_t target_width = 640;
        inline constexpr int32_t target_height = 640;
        inline constexpr int32_t target_channels = 3;
        inline constexpr size_t output_buffer_capacity = 800000;
        inline constexpr size_t inter_op_threads = 1;
        inline constexpr size_t intra_op_threads = 0;
        inline constexpr std::string_view kIphoneStreamUrl = "http://192.168.1.203:8080/stream.mjpeg";
    } // namespace app
} // namespace obj_rec

#endif // APPLICATION_CONSTANTS_H_