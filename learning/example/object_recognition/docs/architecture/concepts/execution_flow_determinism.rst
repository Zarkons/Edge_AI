Execution Flow Determinism
==========================

.. concept:: Execution Flow Determinism
   :id: con_execution_flow_determinism
   :status: draft
   :realizes: adr_single_threaded_cooperative_execution_loop_with_asynchronous_io

   The system guarantees predictable, uniform processing intervals by eliminating operating system thread-scheduling jitter, context-switching overhead, and blocking I/O dependencies from the critical inference path.

.. spec:: Thread Isolation and Cooperative Polling
   :id: spec_thread_isolation_and_cooperative_polling
   :status: draft
   :implements: con_execution_flow_determinism
   :verification_method: Static Analysis
   :verification_criteria: No thread creation or blocking synchronization primitives are present in the core runtime loop translation units.

    The core processing loop executes entirely within a single-threaded cooperative polling architecture.
    Spawning background threads, initializing internal worker pools, or invoking parallel processing libraries within core translation units is strictly prohibited.
    All ingestion, frame alignment, and processing tasks must execute sequentially within the primary execution sequence.

.. spec:: Abstract Non-Blocking Ingress
   :id: spec_abstract_non_blocking_ingress
   :status: draft
   :implements: con_execution_flow_determinism
   :verification_method: Review
   :verification_criteria: Source inspection confirms all ingestion components implement the IInputProvider interface with zero blocking system calls.

    Hardware and network ingress lifecycles are decoupled from the processing loop via a Polymorphic Factory Pattern.
    Concrete ingestion plugins must implement the abstract IInputProvider interface and fetch frame data via non-blocking queries.
    No hardware driver locks or synchronous network polling operations occurs within the streaming execution path.

.. spec:: Zero-Copy IPC Shared Memory Egress
   :id: spec_zero_copy_ipc_shared_memory_egress
   :status: draft
   :implements: con_execution_flow_determinism
   :verification_method: Test
   :verification_criteria: When output is received, it is present in a pre-allocated shared memory segment, and the UI consumer process is notified via a non-blocking OS event flag.

    Data egress completely bypasses high-level messaging queues and heap allocation paths.
    The IOutputDispatcher component streams bounding-box tensors and confidence metrics directly into a pre-allocated OS Shared Memory IPC segment.
    Inter-process notifications are signaled via non-blocking OS event flags, allowing the loop to immediately process the next frame.

.. spec:: Consumer Saturation Frame Dropping
   :id: spec_consumer_saturation_frame_dropping
   :status: draft
   :implements: con_execution_flow_determinism
   :verification_method: Test
   :verification_criteria: When the external GUI process is intentionally frozen, the core pipeline runtime loop continues processing frames at full frequency without blocking or crashing.

    To maintain real-time throughput, the core pipeline never stalls if the independent GUI consumer process falls behind or stops responding.
    The shared memory ring-buffer immediately overwrites old, unconsumed historical frames to prioritize current frame processing latency.
