# third_party/whisper.BUILD
load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

# Define target configurations to mimic their setup flags
config_setting(
    name = "use_bzlmod",
    define_values = {"use_bzlmod": "1"},
)

shared_hdrs = [
    "PmpManager.hpp",
    "PmaManager.hpp",
]

base_hdrs = [
    "HartConfig.hpp",
    "WhisperMessage.h",
    "Hart.hpp",
    "Core.hpp",
    "System.hpp",
    "Server.hpp",
    "Interactive.hpp",
    "DecodedInst.hpp",
    "Decoder.hpp",
    "InstEntry.hpp",
    "InstId.hpp",
    "Isa.hpp",
    "IoDevice.hpp",
    "Uart8250.hpp",
] + shared_hdrs

cc_library(
    name = "shared_headers",
    hdrs = shared_hdrs,
    visibility = ["//visibility:public"],
)

# Replicating their native core library block
cc_library(
    name = "rvcore",
    srcs = glob(
        include = [
            "*.cpp",
            "*.hpp",
            "**/*.cpp",
            "**/*.hpp",
        ],
        exclude = [
            "whisper.cpp",
            "py-bindings.cpp",
            "numa.cpp",  # Bypasses the broken macOS stdlib.h conflict
            "RemoteFrameBuffer.cpp",  # <-- CRITICAL: Drops the unneeded video engine file
            "RemoteFrameBuffer.hpp",  # <-- CRITICAL: Drops the unneeded video engine file
        ] + base_hdrs,
    ),
    hdrs = base_hdrs + glob(
        include = [
            "**/*.h",
            "**/*.hpp",
        ],
        exclude = [
            "RemoteFrameBuffer.hpp",
            "test/**",
        ],
    ),
    copts = [
        "-std=c++20",
        "-O3",
        "-fPIC",
        "-w",  # Suppresses unneeded upstream warnings
        "-Wno-invalid-specialization",  # Unblocks big-integer std overrides on Mac
    ],
    includes = [
        ".",
        "third_party",  # Natively bridges internal JSON and ELFIO subdirectories
        "third_party/softfloat/source/include",
    ],
    local_defines = [
        "SOFT_FLOAT",
        "MEM_CALLBACKS",
        "LZ4_COMPRESS",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":shared_headers",
        "@boost.algorithm//:boost.algorithm",
        "@boost.circular_buffer//:boost.circular_buffer",
        "@boost.format//:boost.format",
        "@boost.program_options//:boost.program_options",
        "@nlohmann_json//:json",
    ],
)

# Replicating their primary entry executable binary target
cc_binary(
    name = "whisper",
    srcs = ["whisper.cpp"],
    linkopts = ["-rdynamic"],
    local_defines = [
        "SOFT_FLOAT",
        "MEM_CALLBACKS",
        "LZ4_COMPRESS",
    ],
    visibility = ["//visibility:public"],
    deps = [":rvcore"],
)
