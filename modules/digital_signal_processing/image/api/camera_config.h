#ifndef DSP_IMAGE_API_CAMERA_CONFIG_H
#define DSP_IMAGE_API_CAMERA_CONFIG_H
#include <string>

namespace dsp::image
{
    enum class CameraType
    {
        IPhoneNetworkStream,
        LocalUSBWebcam,
        FileMockReader
    };

    struct CameraConfig
    {
        CameraType type;
        std::string connection_string; // Holds URL, file path, or device index string
    };
}
#endif // DSP_IMAGE_API_CAMERA_CONFIG_H