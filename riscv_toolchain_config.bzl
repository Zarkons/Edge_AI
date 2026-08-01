load("@rules_cc//cc:action_names.bzl", "ACTION_NAMES")
load("@rules_cc//cc:cc_toolchain_config_lib.bzl", "feature", "flag_group", "flag_set", "tool_path")

# riscv_toolchain_config.bzl
load("@rules_cc//cc/common:cc_common.bzl", "cc_common")

def _impl(ctx):
    compiler_path = "/opt/homebrew/bin/riscv64-unknown-elf-gcc"

    tool_paths = [
        tool_path(name = "gcc", path = compiler_path),
        tool_path(name = "cpp", path = compiler_path),
        tool_path(name = "ar", path = "/opt/homebrew/bin/riscv64-unknown-elf-ar"),
        tool_path(name = "ld", path = "/opt/homebrew/bin/riscv64-unknown-elf-ld"),
        tool_path(name = "nm", path = "/opt/homebrew/bin/riscv64-unknown-elf-nm"),
        tool_path(name = "objcopy", path = "/opt/homebrew/bin/riscv64-unknown-elf-objcopy"),
        tool_path(name = "objdump", path = "/opt/homebrew/bin/riscv64-unknown-elf-objdump"),
        tool_path(name = "strip", path = "/opt/homebrew/bin/riscv64-unknown-elf-strip"),
    ]

    # Ensure ACTION_NAMES.assemble and ACTION_NAMES.preprocess_assemble are imported at the top of your file!
    features = [
        # 1. Instruct Bazel to preserve and append your local copts / flags across C++, C, and Assembly
        feature(
            name = "user_compile_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = ["%{user_compile_flags}"],
                            iterate_over = "user_compile_flags",
                        ),
                    ],
                ),
            ],
        ),
        # 2. Hardcoded layout constraints base rules forced across all compilation/assembly units
        feature(
            name = "default_compile_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = [
                        ACTION_NAMES.c_compile,
                        ACTION_NAMES.cpp_compile,
                        ACTION_NAMES.assemble,
                        ACTION_NAMES.preprocess_assemble,
                    ],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "-Wall",
                                "-std=c++20",
                                "-march=rv32imaf",
                                "-mabi=ilp32f",
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ]

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        toolchain_identifier = "riscv-local-compiler",
        host_system_name = "local",
        target_system_name = "riscv32-unknown-none",
        target_cpu = "riscv32",
        compiler = "gcc",
        abi_version = "unknown",
        tool_paths = tool_paths,
        features = features,  # <--- Active flag routing pass enabled
    )

riscv_cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {},
    provides = [CcToolchainConfigInfo],
)
