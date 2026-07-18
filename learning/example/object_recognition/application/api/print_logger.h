#ifndef PRINT_LOGGER_H_
#define PRINT_LOGGER_H_

#include <iostream>

#if defined(PRINT_LOGGER_ENABLE)
#define PRINT_LOG(...)                                          \
    do                                                          \
    {                                                           \
        std::cout << "[PROFILER] " << __VA_ARGS__ << std::endl; \
    } while (0)
#else
#define PRINT_LOG(...) \
    do                 \
    {                  \
    } while (0)
#endif // PRINT_LOGGER_ENABLE
#endif // PRINT_LOGGER_H_