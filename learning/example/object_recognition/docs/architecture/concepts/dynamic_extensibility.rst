Dynamic Extensibility Boundaries
================================

.. concept:: Dynamic Extensibility Boundaries
   :id: con_dynamic_extensibility_boundaries
   :status: active
   :realizes: adr_dynamic_linkage_polymorphic_plugin_infrastructure

   The system segregates high-speed, statically bound data preparation from volatile downstream execution engines. This ensures maximum pixel-loop compiler optimization while permitting runtime component updates for inference abstractions without shared binary linkage overhead.

.. spec:: Zero-Vtable Preprocessing Linkage
   :id: spec_zero_vtable_preprocessing_linkage
   :status: active
   :implements: con_dynamic_extensibility_boundaries
   :verification_method: Static Analysis
   :verification_criteria: Compilation or link-time gates fail if image preprocessing class declarations contain the 'virtual' keyword or generate dynamic dispatch symbols.

    Image preprocessing modules compile strictly as concrete types or compile-time templates like CRTP.
    These libraries couple directly to the core binary via standard compile-time and link-time flags.
    This explicit binding allows full compiler function inlining and prevents vtable memory-hopping overhead.
    The hot-path pixel manipulation loops remain completely uninhibited by dynamic dispatch latency.

.. spec:: Pure Virtual Inference Abstraction
   :id: spec_pure_virtual_inference_abstraction
   :status: active
   :implements: con_dynamic_extensibility_boundaries
   :verification_method: Compile-Time Check
   :verification_criteria: Compilation fails via static_assert size-matching tests if inference interface definitions contain internal state variables or non-virtual method declarations.

    Volatile downstream execution frameworks inherit exclusively from pure virtual C++ interface structures.
    These interface classes maintain a zero-state configuration across the dynamic binary boundaries.
    This design prevents the accumulation of unaligned runtime states or concrete layout mismatches.
    Polymorphic behavior relies purely on standard virtual table lookups triggered only at the frame boundary.

.. spec:: Decoupled Shared Inference Library Extraction
   :id: spec_decoupled_shared_inference_library_extraction
   :status: active
   :implements: con_dynamic_extensibility_boundaries
   :verification_method: Static Analysis
   :verification_criteria: Core orchestration link gates fail if the core engine contains direct compile-time reference symbols to external inference plugin implementation objects.

    Inference modules compile into standalone dynamic shared libraries isolated from core binary linking flags.
    The primary execution core resolves and routes data through these libraries purely at execution time.
    The system maps only the specific shared libraries demanded by the active profile configuration into process memory.
    Heavy vendor-specific deep learning SDK footprints remain completely decoupled from the core baseline executable.

.. spec:: C-Linkage Factory Gateway
   :id: spec_c_linkage_factory_gateway
   :status: active
   :implements: con_dynamic_extensibility_boundaries
   :verification_method: Static Analysis
   :verification_criteria: Global symbol tables for compiled plugin binaries contain the unmangled extern "C" symbols create_plugin and destroy_plugin.

    Every inference plugin shared object exposes standard, unmangled export hooks to guarantee clean symbol resolution.
    A dedicated ``extern "C"`` factory gateway instantiates and safely tears down localized plugin contexts.
    The core pipeline interacts with these gateways exclusively via native operating system dynamic loading utilities.
    This flat binary gateway pattern bypasses C++ name-mangling issues across differing compiler versions.

.. spec:: Isolated Initialization Linkage
   :id: spec_isolated_initialization_linkage
   :status: active
   :implements: con_dynamic_extensibility_boundaries
   :verification_method: Test
   :verification_criteria: Dynamic library discovery, handle allocation, and symbol resolution actions must finish before the first active frame enters the live loop.

    The system confines all dynamic library discovery, handle allocation, and symbol resolution actions entirely to the boot phase.
    The runtime pipeline parses the active profile configuration and maps the required inference plugin files into process memory.
    It resolves all function pointers completely before ingesting live sensor or camera streams.
    This initialization boundary locks down polymorphic pathways, keeping the execution loop entirely free from dynamic linking latency.
