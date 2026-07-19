Performance Profiling
---------------------

Following figure shows initial performance measurement of the object recognition pipeline, which is not optimized and runs on a single thread. The inference and preprocessing are done sequentially, which results in lower throughput and higher latency.

.. figure:: /docs/performance_measurements/perf_initial_measurement.png
   :align: center
   :alt: First performance measurement of the object recognition pipeline.

Following figure shows optimized performance measurement of the object recognition pipeline, which runs on a single thread. The inference and preprocessing are done sequentially, but the code is optimized for better performance, minimal memory copies, and dynamic allocation in hot path. Still single-threaded execution with no specific hardware acceleration is used, which results in better throughput and lower latency compared to the initial measurement.

.. figure:: /docs/performance_measurements/perf_single_thread.png
   :align: center
   :alt: Single-threaded optimized inference and preprocessing performance measurement of the object recognition pipeline.

Following figure shows optimized performance measurement of the object recognition pipeline, which runs on two threads. The inference and preprocessing are done in parallel with the reading the frames from the camera, which results in better throughput and lower latency compared to the single-threaded execution.

.. figure:: /docs/performance_measurements/perf_worker_thread.png
   :align: center
   :alt:  Optimized inference and preprocessing with worker thread that reads frames from the camera.