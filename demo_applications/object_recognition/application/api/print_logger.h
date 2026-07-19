#ifndef PRINT_LOGGER_H_
#define PRINT_LOGGER_H_

#include <iostream>

#if defined(PRINT_LOGGER_ENABLE) || defined(PRINT_LOGGER_INFO_ENABLE)
#define PRINT_INFO(...)                                     \
    do                                                      \
    {                                                       \
        std::cout << "[INFO] " << __VA_ARGS__ << std::endl; \
    } while (0)
#else
#define PRINT_INFO(...) \
    do                  \
    {                   \
    } while (0)
#endif // PRINT_LOGGER_INFO_ENABLE

#if defined(PRINT_LOGGER_ENABLE) || defined(PRINT_LOGGER_ERROR_ENABLE)
#define PRINT_ERROR(...)                                     \
    do                                                       \
    {                                                        \
        std::cerr << "[ERROR] " << __VA_ARGS__ << std::endl; \
    } while (0)
#else
#define PRINT_ERROR(...) \
    do                   \
    {                    \
    } while (0)
#endif // PRINT_LOGGER_ERROR_ENABLE
#endif // PRINT_LOGGER_H_