#ifndef SAMPLE_PROCESSING_FROM_FILE_H_
#define SAMPLE_PROCESSING_FROM_FILE_H_

#include <string>
#include <vector>
#include "common_types.h"

namespace audio_sample_processing
{
    struct AudioSample
    {
        std::string path;
        ml_types_dynamic::RawSignalFloatVec raw_waveform;
    };

    AudioSample GetSample();

} // namespace audio_sample_processing

#endif // SAMPLE_PROCESSING_FROM_FILE_H_