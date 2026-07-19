#pragma once
#include <iostream>
#include <mutex>
#include <cstring> // Required for strlen()

#if defined(TRACY_ENABLE)
#include <Tracy.hpp>

namespace tracy_internal
{
    struct ScopedLockProfile
    {
        // Order matters: unique_lock must be initialized AFTER lock_zone
        // so that the Tracy zone captures the exact moment the thread begins waiting.
        tracy::ScopedZone lock_zone;
        std::unique_lock<std::mutex> lock;

        ScopedLockProfile(std::mutex &mtx, const char *name, const char *function, const char *file, int line)
            : lock_zone(
                  static_cast<uint32_t>(line),
                  file,                  // Source file
                  std::strlen(file),     // Source file size
                  function,              // Function name
                  std::strlen(function), // Function name size
                  name,                  // Custom Lock zone name (e.g., "Worker Buffer Lock")
                  std::strlen(name),     // Custom Lock zone name size
                  0,                     // Color (0 = default)
                  -1,                    // Depth
                  true                   // Is Active
                  ),
              lock(mtx, std::defer_lock)
        {
            // The Tracy zone is already open, recording wait-state time.
            // This line blocks the CPU core until the mutex is acquired.
            lock.lock();
        }
    };
}

// Pass local compiler definitions safely into our structural wrapper constructor
#define LockGuardType(mtx, name) \
    tracy_internal::ScopedLockProfile __tracy_lock_instance(mtx, name, __FUNCTION__, __FILE__, __LINE__)

#define PROFILER_LOG(...)                                       \
    do                                                          \
    {                                                           \
        std::cout << "[PROFILER] " << __VA_ARGS__ << std::endl; \
    } while (0)
#else
// Production fallback configuration block
#define LockGuardType(mtx, name) std::lock_guard<std::mutex> __std_lock_instance(mtx)

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
