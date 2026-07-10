2. Constraints
===================

.. req:: Model Size
    :id: req_model_size
    :status: draft
    :req_type: constraint

        The model size shall not exceed 5 MB to ensure it can be deployed on resource-constrained edge devices.

.. req:: Build System
    :id: req_build_system
    :status: draft
    :req_type: constraint

        The system shall be built using the Bazel build system to ensure consistent builds across different environments.

.. req:: Target Environments
    :id: req_target_environments
    :status: draft
    :req_type: constraint

        The system must run natively across two distinct host environments:
            -  A Mac platform (Apple Silicon M-series ARM64 architecture running macOS).
            -  A Raspberry Pi 5 Single Board Computer (Broadcom ARM64 running QNX/Linux/Raspberry Pi OS).

.. req:: Implementation Language
    :id: req_lang_constraint
    :status: draft
    :req_type: constraint

        The core pipeline and application logic must be written exclusively in C++20 or Python 3.11+ to align with Bazel toolchains on both macOS and Linux.    
