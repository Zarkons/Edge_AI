2. Constraints
===================

.. constraint:: Model Size
    :id: const_model_size
    :status: draft

    The model size shall not exceed 5 MB to ensure it can be deployed on resource-constrained edge devices.

.. constraint:: Build System
    :id: const_build_system
    :status: draft

    The system shall be built using the Bazel build system to ensure consistent builds across different environments.

.. constraint:: Target Environments
    :id: const_target_environments
    :status: draft

    The system must run natively across two distinct host environments:
        -  A Mac platform (Apple Silicon M-series ARM64 architecture running macOS).
        -  A Raspberry Pi 5 Single Board Computer (Broadcom ARM64 running QNX/Linux/Raspberry Pi OS).

.. constraint:: Implementation Language
    :id: const_lang_constraint
    :status: draft

    The core pipeline and application logic must be written exclusively in C++20 or Python 3.11+ to align with Bazel toolchains on both macOS and Linux.    
