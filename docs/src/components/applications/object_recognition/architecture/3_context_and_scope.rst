3. Context And Scope
====================


.. mermaid:: diagrams/black_box_system_view.mmd
   :align: center
   :caption: Black Box View of the Object Recognition System


.. list-table:: Business Context Boundaries
   :widths: 20 20 60
   :header-rows: 1

   * - Domain Neighbor
     - Direction
     - Description / Business Responsibility
   * - **CAM**
     - Inbound
     - Captures and feeds live environmental image observations for real-time item tracking
   * - **FILE**
     - Inbound
     - Provides an archival image storage database for batch-processing pre-recorded video sequences
   * - **User Interface**
     - Outbound
     - Collects and visualizes real-time bounding box locations and classification confidence maps
   * - **Telemetry Collector**
     - Outbound
     - Tracks diagnostic engine health data to enable out-of-band monitoring performance auditing
   * - **Inference Subsystem**
     - Bi-directional
     - Evaluates mathematical matrix representations of images to yield structural categorization coordinates
   * - **Model Asset Resource**
     - Inbound
     - Provides the pre-trained neural network weights and graph structure required to classify objects
    
.. list-table:: Technical Context Interfaces
   :widths: 20 30 50
   :header-rows: 1

   * - Technical Neighbor
     - Interfaces / Transfer Protocol
     - Engineering Description & Data Envelopes
   * - **CAM**
     - URL scheme: ``http://<camera_ip_address>/<endpoint>``
     - Ingests raw frames continuously via HTTP GET requests utilizing standard multipart MJPEG streaming payload data blocks.
   * - **FILE**
     - Native POSIX Filesystem Stream Read
     - Mounts local or networked block storage directories to open static encoded images formatted as raw JPEG or PNG files.
   * - **User Interface**
     - Thread-Safe Shared Memory Ring
     - Dispatches calculated classification strings, floating-point confidence vectors, and integer bounding box arrays out-of-band to prevent UI render blocking.
   * - **Telemetry Collector**
     - Asynchronous TCP Socket Streaming
     - Pipes low-overhead execution timestamps and instrumentation markers over a dedicated network port to an out-of-band diagnostic application.
   * - **Inference Subsystem**
     - Abstract Virtual C++ Interface
     - Core execution branch interacts with a tool-agnostic C++ Virtual Function Table (VTable API), passing continuous flat float matrix arrays to decouple domain logic from specific vendor implementations.
   * - **Model Asset Resource**
     - Static Resource Embedding
     - Loads a pre-trained serialized neural network graph asset file from a local storage pathway during initial application startup routines.
