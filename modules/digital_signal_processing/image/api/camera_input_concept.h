#ifndef DSP_IMAGE_API_CAMERA_INPUT_CONCEPT_H
#define DSP_IMAGE_API_CAMERA_INPUT_CONCEPT_H
#include <concepts>
#include "dip_data_types.h"
#include "camera_config.h"

namespace dsp::image
{
    template <typename T>
    concept CameraInputHandler = requires(T handler, const CameraConfig &config, VideoFrame &frame) {
        { handler.Initialize(config) } -> std::same_as<bool>;
        { handler.GetNextFrame(frame) } -> std::same_as<bool>;
    };
}
#endif // DSP_IMAGE_API_CAMERA_INPUT_CONCEPT_H