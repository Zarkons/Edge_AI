#ifndef SAMPLE_PROCESSING_H
#define SAMPLE_PROCESSING_H

#include <string>
#include <vector>
#include "common_types.h"

using namespace MlTypesDynamic;

struct AudioSample
{
    std::string path;
    RawSignalFloatVec raw_waveform;
};

AudioSample GetSample();
std::vector<std::string> GetLabelNamesList();

#endif // SAMPLE_PROCESSING_H