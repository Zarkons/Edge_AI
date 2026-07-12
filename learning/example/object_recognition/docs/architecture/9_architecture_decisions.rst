9. Architecture Decisions
=========================

.. adr:: Low-Latency Native Runtime Standardization
    :id: adr_low_latency_native_runtime_standardization
    :status: draft
    :enables: str_native_compiled_pipeline_core

    .. rubric:: Context

    Real-time computer vision inference loops running on edge environments require predictable execution speeds with zero overhead from language runtimes. High-level interpreted languages (like Python) introduce severe multi-threading bottlenecks via interpreter locks, while garbage-collected runtimes (like Go or Java) cause non-deterministic processing pauses during memory reclamation cycles. To achieve the sub-millisecond precision required by our quality goals, the system must execute as raw, optimized machine instructions directly on the bare target CPU hardware.

    .. rubric:: Decision

    Enforce ISO C++20 as the exclusive language standard for the entire core processing pipeline. Disable RTTI (``-fno-rtti``) and enforce high compiler optimizations (``-O3``, ``-flto``).

    **Language Specification Boundaries**
    
    * **Standard Selection**: All system core components must compile cleanly under the ``-std=c++20`` flag. Language features that trigger hidden heap allocations behind the scenes (such as unconfigured coroutines or raw ranges) are banned from the processing loop.
    * **Runtime Metadata Removal**: Run-Time Type Information is completely stripped via ``-fno-rtti`` to eliminate type-description binary bloat and prevent non-deterministic downcasting performance penalties.

    **Compiler Optimization Toolchain**
    
    The target build definitions mandate aggressive optimizations (``-O3``) combined with Link-Time Optimization (``-flto``). This permits the compiler to perform deep cross-module inlining, vector loop transformations, and comprehensive dead-code elimination across separate source units.

    .. rubric:: Consequences

    **Positive Consequences (Benefits)**

    * **Deterministic Pipeline Processing**: Eliminates language-level garbage collection pauses, interpreter engine startups, and runtime type checks, delivering true bare-metal performance.
    * **Significant Binary Footprint Compression**: Stripping RTTI data tables and enabling link-time dead-code pruning noticeably shrinks the final executable file size for memory-restricted storage blocks.
    * **Modern Zero-Cost Abstractions**: Leveraging C++20 compile-time concepts and ``constexpr`` mechanics provides clean, readable code architecture with zero runtime execution penalties.

    **Negative Consequences (Risks and Mitigations)**

    * **Complete Loss of Dynamic Downcasting**: Developers cannot use ``dynamic_cast`` or the ``typeid`` operator, requiring polymorphic plugin designs to rely purely on fast compile-time contracts or virtual table method lookups.
    * **Manual Memory Management Discipline**: Relinquishing a managed runtime means coding errors can result in immediate segmentation faults rather than catchable exceptions, demanding rigorous static analysis testing within target pipelines.
    * **Elevated C++ Standardization Profile**: Engineering teams must be deeply proficient in modern C++ patterns, move semantics, and target toolchain compilation optimization rules.


.. adr:: Zero-Dynamic Allocation in Hot-Path
    :id: adr_zero_dynamic_allocation_in_hot_path
    :status: draft
    :enables: str_zero_allocation_runtime_loop

    .. rubric:: Context

    The core object recognition system executes on resource-constrained edge hardware under strict real-time constraints. In standard C++, invoking dynamic heap allocation operations (such as ``new``, ``delete``, ``malloc``, or resizing standard container classes) forces context switches into the kernel's memory manager. These operations have non-deterministic execution times due to heap-searching, thread locks, and background fragmentation cleanup. Over extended operational windows, localized heap fragmentation can lead to unpredictable latency spikes or sudden out-of-memory (OOM) kernel panics.

    .. rubric:: Decision

    Forbid all dynamic heap memory allocations (``malloc``, ``free``, ``new``, ``delete``) during the active streaming inference phase. Pre-allocate all data arenas during initialization.

    **Lifecycle Phase Restrictions**
    
    * **Initialization Phase**: Dynamic heap memory configurations and allocations are explicitly permitted to establish the primary system context, construct plugins via factories, and secure memory segments.
    * **Active Inference Phase**: Once the single-threaded cooperative processing loop ingests its first frame, the invocation of any operation that relies on implicit or explicit heap allocations is strictly prohibited.

    **Enforcement Controls**
    
    The architecture forces implementation compliance by modifying target compilation toolchain configurations to hook, flag, and fail on runtime allocations, backed by strict ``clang-tidy`` static analysis validation rules.

    .. rubric:: Consequences

    **Positive Consequences (Benefits)**

    * **True Timing Determinism**: Eliminates heap allocation latency spikes entirely, bounding processing cycles strictly to mathematical execution speeds.
    * **Elimination of Runtime Fragmentation**: The heap remains completely untouched after setup, ensuring absolute system stability and avoiding OOM crashes during long-term field deployment.
    * **Enhanced Cache Efficiency**: Utilizing pre-allocated contiguous memory pools maximizes CPU L1/L2 cache locality, accelerating tensor preparation loops.

    **Negative Consequences (Risks and Mitigations)**

    * **Rigid Operational Capacity Ceilings**: The system must declare its maximum resource limits at startup; if an input frame resolution exceeds pre-allocated bounds, it must be rejected or cropped.
    * **Increased Development Overhead**: Engineering teams lose the convenience of dynamic standard library collections, requiring all internal algorithmic data structures to use fixed arrays, stacks, or static buffer blocks.


.. adr:: Single-Threaded Cooperative Execution Loop with Asynchronous I/O
    :id: adr_single_threaded_cooperative_execution_loop_with_asynchronous_io
    :status: draft
    :enables: str_io_boundary_abstraction

    .. rubric:: Context

    The core object recognition pipeline must process high-frequency video frames under strict real-time edge constraints. Traditional multi-threaded rendering, blocking hardware I/O operations, and dynamic memory messaging queues introduce unpredictable context-switching delays, CPU cache eviction issues, and latency spikes. To maintain absolute timing determinism, the critical execution core must be fully isolated from external GUI lifecycles, network volatility, and device connection drops.

    .. rubric:: Decision

    Enforce a strict single-threaded, cooperative execution loop for the primary core system lifecycle; the core processing binary must never spawn internal background threads. To isolate the execution logic from hardware and external component lifecycles, concrete ingress providers must be instantiated at startup via a Polymorphic Factory Pattern matching the abstract IInputProvider interface. Data egress must be isolated by having the IOutputDispatcher write bounding-box tensors and confidence metrics directly into a pre-allocated OS Shared Memory (POSIX / SysV IPC) segment. Once written, the loop triggers a non-blocking OS event flag and immediately continues processing. A completely separate, dedicated consumer process handles reading from this shared memory segment to execute post-processing and forward results to the GUI.

    .. rubric:: Consequences

    **Positive Consequences (Benefits)**

    * **Complete I/O Isolation**: The critical inference pipeline remains highly responsive and free from blocking operations since all data ingress and egress are completely decoupled from external device lifecycles.
    * **Jitter-Free Determinism**: Utilizing a single-threaded cooperative polling engine eliminates OS thread scheduling overhead, keeping latency predictably uniform across prolonged execution periods.
    * **Zero-Copy High Throughput**: Writing directly to fixed, pre-allocated shared memory segments entirely bypasses heap allocation paths, permitting maximum data throughput with zero transmission lag.
    * **Clean Structural Decoupling**: Forcing separate process boundaries isolates heavy GUI component lifecycles or application rendering crashes from halting the core object recognition core.

    **Negative Consequences (Risks and Mitigations)**

    * **Elevated IPC Structural Complexity**: Shifting to multi-process communication requires low-level architecture tooling to orchestrate native POSIX/SysV handles and layout precise shared memory structures.
    * **Strict No-Blocking Implementer Discipline**: Developers are strictly prohibited from embedding blocking operations within the plugins; a single slow system call will block the entire application process loop.
    * **Vulnerability to Consumer Saturation**: If the independent GUI consumer process stalls or drops below the core's frame rate, shared memory slots could overwrite unconsumed historical frames to preserve real-time throughput.


.. adr:: ONNX Runtime Framework and Quantization
    :id: adr_onnx_runtime_framework_and_quantization
    :status: draft
    :enables: str_edge_quantization_pruning, str_model_agnostic_input_processing

    .. rubric:: Context

    The core object recognition loop executes on resource-constrained edge environments. 
    To maintain deterministic execution speeds and fit within our strict pre-allocated 
    memory boundaries, neural network models cannot be deployed in heavy, raw training formats. 
    We require an abstraction layer that compresses the model’s memory footprint and 
    utilizes hardware-accelerated integer vector instructions to maximize throughput.

    .. rubric:: Decision

    We standardize on the ONNX Runtime (Native C++ API) as our uniform inference engine 
    wrapper and mandate an ahead-of-time INT8 Quantization pipeline for all deployed models.

    **Core Pipeline Linkage**

    The core pipeline will compile directly against the dynamic native C++ 
    ONNX Runtime libraries. Direct runtime dependencies on high-level Python wrappers, 
    PyTorch runtimes, or heavy framework inference setups are strictly forbidden.

    **Offline Optimization Rules**

    Models must pass through an offline, automated compilation script before deployment. 
    This optimization pipeline must perform:

    * **Structural Pruning**: Remove redundant neural nodes and unneeded graph layers.
    * **Quantization**: Apply Post-Training Quantization (PTQ) or Quantization-Aware 
      Training (QAT) to convert weights and activations from standard 32-bit 
      floating-point (``FP32``) to calibrated 8-bit integers (``INT8``).

    **Target Architecture Configurations**

    The runtime initialization phase will dynamically query the hosting platform and 
    bind the model to the target hardware’s optimal native Execution Provider (EP) 
    through decoupled configurations:

    * **Linux / Raspberry Pi 5 (Production)**: Register the default CPU Execution Provider. 
      The cross-compilation toolchain must pass native target architecture flags to map 
      quantized integer operations directly to hardware-accelerated ARM NEON SIMD vector registers.
    * **Apple Silicon / macOS (Testing & Evaluation)**: Register the default CPU Execution Provider. 
      This bypasses specialized NPU compilation steps, guarantees exact numerical behavior parity 
      with the target edge device, and ensures instantaneous model loading during test execution loops.

    **Memory Bounds and Allocations**

    We will utilize the ONNX Runtime Custom Memory Allocator / IOBinding API to map the input 
    and output tensors directly into the pre-allocated arenas. The inference engine is forbidden 
    from allocating temporary heap buffers during the execution loop.

    .. rubric:: Consequences

    **Positive Consequences (Benefits)**

    * **Training Framework Agnosticism**: The core C++ pipeline decouples completely from 
      data science infrastructure; it remains strictly unconcerned with whether a model 
      originated in PyTorch, TensorFlow, or JAX, requiring only a standard compliance-validated 
      ``.onnx`` file.
    * **Drastic Memory Bandwidth Reduction**: Compressing weights from ``FP32`` to ``INT8`` 
      lowers cache and memory bandwidth consumption by exactly 75%. This enables resource-constrained 
      edge CPUs to maximize hardware-accelerated NEON SIMD vector instruction saturation.
    * **Heap-Free Execution Alignment**: Leveraging ONNX Runtime’s custom ``IOBinding`` API 
      passes raw pointers directly between pre-allocated image arenas and target model tensors. 
      This eliminates implicit heap allocations and ensures compliance with our strict 
      zero-allocation runtime policy.

    **Negative Consequences (Risks and Mitigations)**

    * **Numerical Precision Degradation**: Dropping floating-point precision down to 8-bit integers 
      introduces localized quantization noise. Every unique model revision must undergo automated 
      CI/CD evaluation against validation datasets to certify that confidence metrics do not slip.
    * **Pipeline-Induced Deployment Friction**: Data science teams lose the capability to deploy 
      raw, uncompressed training checkpoints directly onto target systems. Teams must maintain 
      and actively calibrate an ahead-of-time optimization toolchain to output compliant, 
      calibrated artifacts.

.. adr:: Dynamic Linkage Polymorphic Plugin Infrastructure
    :id: adr_dynamic_linkage_polymorphic_plugin_infrastructure
    :status: draft
    :enables: str_polymorphic_runtime_plugins

    .. rubric:: Context

    The edge object recognition core must adapt to evolving computer vision models, new pre-processing algorithms, and different hardware acceleration modules over time. Hardcoding these processing steps directly into the core execution binary forces complete system recompilations for minor logic updates, increasing software distribution overhead and testing regression cycles. To achieve target flexibility goals, the system must separate the high-speed orchestration engine from localized execution algorithms through an expandable runtime infrastructure.

    .. rubric:: Decision

    Implement a decoupled plugin framework utilizing standard C++ pure virtual interface definitions loaded at runtime via native operating system dynamic loading utilities (``dlopen``/``dlsym`` on Linux/QNX/macOS).

    **Interface and Linkage Architecture**
    
    * **State-Free Interfaces**: All extensible components must inherit from zero-state, pure virtual C++ class definitions. 
    * **Dynamic Shared Objects**: Processing modules must be compiled as independent, standalone dynamic shared libraries (``.so`` or ``.dylib``) stored in a dedicated repository sub-folder.
    * **C-Linkage Gateway Functions**: To avoid C++ name-mangling issues across differing compiler versions, every plugin shared object must expose a standard ``extern "C"`` factory gateway function (such as ``create_plugin`` and ``destroy_plugin``) for the system core to hook into.

    .. rubric:: Consequences

    **Positive Consequences (Benefits)**

    * **Isolated Runtime Extensibility**: New inference algorithms or custom image manipulation blocks can be dynamically swapped in or out via configuration profiles without modifying or recompiling the core processing pipeline.
    * **Granular Team Velocity**: Engineering teams can develop, test, and release separate analytical components as independent versioned binaries, reducing deployment pipeline bottlenecks.
    * **Minimized Active Executable Footprint**: The core engine only maps the specific shared objects demanded by the active operational pipeline profile into system memory, avoiding unneeded binary loading.

    **Negative Consequences (Risks and Mitigations)**

    * **Fragile Application Binary Interface (ABI) Vulnerability**: Compiling individual plugins with mismatched toolchain versions can lead to memory alignment crashes. The deployment pipeline must strictly enforce matching compiler baselines and inject strict semantic version checking macros into the plugin discovery phase.
    * **Initialization-Phase Overhead**: Dynamic linking introduces localized system latency during boot cycles while loading symbols; this is accepted since it remains confined entirely to the initialization phase.

.. adr:: Lock-Free Atomic Queue Telemetry Engine
    :id: adr_lock_free_atomic_queue_telemetry_engine
    :status: draft
    :enables: str_non_blocking_telemetry_logging

    .. rubric:: Context

    Comprehensive runtime observability is mandatory to evaluate system health, frame drop rates, and model confidence metrics on field devices. However, standard logging libraries write directly to disk files or network sockets, introducing unpredictable kernel blocking states and heavy system I/O latency. Inside a high-speed, single-threaded execution loop, a single blocking write would destroy real-time processing guarantees, directly compromising the determinism of the computer vision core.

    .. rubric:: Decision

    Standardize on **Tracy Profiler** as the core native telemetry and system performance logging framework. The system will leverage Tracy's internal lock-free, zero-allocation micro-profiling infrastructure to capture runtime traces.

    **Telemetry Integration Rules**
    
    * **Hot-Path Instrumentation**: The core single-threaded loop will be instrumented using standard Tracy macros (such as ``ZoneScoped`` or ``TracyPlot``). These macros perform non-blocking, atomic pointer writes into Tracy's internal high-speed memory buffers.
    * **Telemetry Separation**: The main execution thread will never touch network or disk I/O for logging. Tracy's internal background worker thread will manage the async collection and broadcasting of trace frames across the network socket to the Tracy server UI.
    * **Conditional Production Compilation**: Tracy instrumentation must be bound to a specific compile target definition (e.g., ``TRACY_ENABLE``). For strict production deployments where network telemetry is unneeded, the macros will compile out to absolute zero-cost null statements.

    .. rubric:: Consequences

    **Positive Consequences (Benefits)**

    * **Production-Grade Zero-Impact Profiling**: High-frequency zone tracking, frame timings, and memory snapshots are captured with sub-microsecond overhead, causing zero timing distortion inside the critical hot-path.
    * **No Custom Queue Wheels**: Eliminates the maintenance overhead of writing a custom lock-free ring buffer or multi-process IPC serialization engine since Tracy handles it natively.
    * **Rich Visual Observability**: Unlocks deep, real-time visual analysis of frame rates, CPU core usage, cache misses, and pipeline bottlenecks without modifying application code structures.

    **Negative Consequences (Risks and Mitigations)**

    * **Background Memory Consumption**: Tracy maintains an internal, fixed-size atomic ring buffer to hold trace events. If the network connection to the server drops, memory usage will grow up to a declared limit, which must be carefully constrained to prevent running out of space on edge devices.
    * **External Dependency Risk**: Integrating Tracy requires importing its native source files into the repository. The build system must isolate this dependency strictly within a specific profiling target to ensure core code stays clean.

.. adr:: Hermetic Multi-Platform Toolchain Matrix Layout
    :id: adr_hermetic_multi_platform_toolchain_matrix_layout
    :status: draft
    :enables: str_cross_platform_build_system, str_multi_architecture_support

    .. rubric:: Context

    Building code for varying edge environments—such as Raspberry Pi 5 running embedded Linux or macOS workstations—typically introduces environmental drift. To realize a cross-platform compilation pipeline under strict real-time hardware criteria, we cannot rely on variable host system states. Historically, supporting multiple target environments forces developers to inject brittle preprocessor conditional blocks (``#ifdef __APPLE__``) or manually switch compiler flags. This practice fractures code logic, degrades maintainability, and allows platform compilation errors to leak past local testing loops.

    .. rubric:: Decision

    Standardize on a centralized, declarative **Platforms and Toolchains Matrix** within the build configuration files to orchestrate all cross-platform and multi-architecture targets.

    **Cross-Platform Boundary Rules**
    
    * **Centralized Mappings**: All deployment hardware architectures and target operating systems must be explicitly declared using native target matching definitions (such as ``constraint_value`` and ``platform``) inside a root-level configurations package (e.g., ``//platforms``).
    * **Decoupled Flag Injection**: Application source code blocks (``cc_library`` configurations) are strictly forbidden from hardcoding raw compilation flags or target architecture switches. All environment-specific optimization flags must switch dynamically through target selection tables (``select()`` mappings).

    **Multi-Architecture Segregation Rules**
    
    * **Banned Preprocessor Toggles**: The use of preprocessor compiler macros to switch core operational compilation logic based on the target architecture is banned inside domain source files.
    * **Hermetic Toolchain Registration**: The repository root must encapsulate explicit cross-compilation toolchain matrices (configuring explicit ``cc_toolchain`` rules). These configurations must tightly bind target compilers, sysroots, and target architectures to their matching destination platforms inside the main configuration space.

    .. rubric:: Consequences

    **Positive Consequences (Benefits)**

    * **Absolute Reproducibility and Parity**: Moving compilation rules entirely into declarative Starlark schemas guarantees bit-identical binary generation across different platforms, local setups, and remote automated builders.
    * **Deterministic Target Switching**: Compiling the entire system for an alternative destination environment is reduced to passing a single command-line target platform flag parameter, automatically re-mapping all optimization flags and compiler paths.
    * **Pristine Source Segregation**: Eliminates the need to pollute application source code with build-system logic or platform detection scripts, preserving clean module boundaries.

    **Negative Consequences (Risks and Mitigations)**

    * **Elevated Platform Boilerplate Overhead**: Writing explicit target constraint profiles and configuration matrices introduces more initial design and file tracking overhead than legacy procedural configurations.
    * **Cross-Toolchain Workspace Complexity**: Integrating independent cross-compilers and isolated sysroots inside a declarative layout requires precise up-front configuration to maintain clean toolchain isolation boundaries.
