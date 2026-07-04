#pragma once
#include <iostream>

#if defined(TRACY_ENABLE)
#include <Tracy.hpp>
#define PROFILER_LOG(...)                                       \
    do                                                          \
    {                                                           \
        std::cout << "[PROFILER] " << __VA_ARGS__ << std::endl; \
    } while (0)
#else
#define PROFILER_LOG(...) \
    do                    \
    {                     \
    } while (0)
#define ZoneScopedN(name) \
    do                    \
    {                     \
    } while (0)
#define ZoneScoped \
    do             \
    {              \
    } while (0)
#define FrameMark \
    do            \
    {             \
    } while (0)
#endif