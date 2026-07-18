Tools
=====

This section provides information about the tools used in the project.

Build System
------------
The Edge AI project uses Bazel as its build system. Bazel is used to build the project, run tests, and generate documentation.
Developer needs to have Bazel installed on their system to build the project. The rest of the tool dependencies are automatically downloaded and built during the build process, and handled by Bazel. No virtual environment is needed, and no additional dependencies are required to be installed on the system.

Currently supported platforms are:
    - MacOS (minimum version 14.0)

.. note::
    The project is not tested on other platforms, and it may not work on other platforms. However, it is expected to work on other platforms as well, as long as Bazel is supported on that platform, with potentially some minor modifications to the build files.

Testing Framework
-----------------
The project uses Google Test (gtest) as the testing framework.

Documentation Generation
------------------------
The project uses Sphinx to generate documentation. The documentation is written in reStructuredText (reST) format, together with doxygen for API documentation, and can be built using the provided Bazel build targets. The overall documentation structure does not follow any specific format, but dedicated component documentation is described using Arc42 template, which is a template for architecture documentation. The Arc42 template provides a structured way to document the architecture of software systems, including their components, relationships, and design decisions. For more information about the Arc42 template, please refer to the official Arc42 website: https://arc42.org/

The derived template in sphinx can be found in :doc:`/arch42_template/arch42_template`


Performance Profiling
---------------------

This project uses Tracy as a performance profiling tool. Tracy is a real-time, nanosecond resolution, remote telemetry, hybrid frame and sampling profiler.