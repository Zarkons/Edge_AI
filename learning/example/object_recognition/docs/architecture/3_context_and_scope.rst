3. Context And Scope
====================


.. mermaid:: diagrams/black_box_view.mmd
   :align: center
   :caption: Black Box View of the Object Recognition System


.. list-table:: Business Context
   :widths: 25 25 50
   :header-rows: 1

   * - Dependency
     - Direction
     - Description
   * - CAM
     - Inbound
     - Captures live images for processing.
   * - FILE
     - Inbound
     - Provides image database for processing.
   * - DSP:Image
     - Inbound
     - Provides data types and class blueprints required for image processing.
   * - ONNX Runtime
     - Bi-directional
     - Passes tensor inputs and receives model bounding box inferences.

    
.. list-table:: Technical Context
   :widths: 25 25 50
   :header-rows: 1

   * - Dependency
     - Interfaces
     - Description
   * - CAM
     - URL scheme: http://<camera_ip_address>/<endpoint>
     - Provides live video stream via HTTP GET requests. The camera is expected to support MJPEG video streaming.
   * - FILE
     - File system access
     - Provides access to a local or networked file system containing images for processing. The system should support common image formats such as JPEG and PNG.
   * - DSP:Image
     - :cpp:any:`dsp::image`
     - Provides a set of C++ class blueprints for image processing, including image loading, resizing, and format conversion.
