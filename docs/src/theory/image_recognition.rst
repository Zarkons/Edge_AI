Image Recognition
=================

This document details the universal image processing and mathematical foundations of image recognition. It covers the fundamental concepts, structures, and optimizations required to represent images in computing systems.

Image Representation
--------------------

Raw images are represented as a matrix of pixels, where each pixel encodes color intensity data. The standard format is the **RGB (Red, Green, Blue) model**, where three independent color channels determine the final pixel appearance.

Layout Specifications
~~~~~~~~~~~~~~~~~~~~~~~

* **Data Structure**: Multi-dimensional array (Tensor).
* **Channel Depth**: Typically 8-bit unsigned integers (values 0-255) per channel, totaling 24 bits per pixel.
* **Storage Order**: Row-major format.

Dimensions and Memory Locality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Image arrays are indexed by three dimensions in a strict hierarchy:

.. code-block:: text

   Matrix Shape: [Height, Width, Channels]

For example, a standard 640×480 RGB image maps to an array of size ``[480, 640, 3]``.

* **First Dimension (Height)**: Represents the total number of rows.
* **Second Dimension (Width)**: Represents the total number of columns.
* **Third Dimension (Channels)**: Represents the color components.

This ordering leverages the **cache locality principle**. Contiguous memory addresses store adjacent pixels in the same row, maximizing CPU/GPU cache hits during sequential row-major processing loops.


Stride and Padding
~~~~~~~~~~~~~~~~~~~

Another critical concept in image representation is **stride** (also known as pitch). Stride defines the total number of bytes between the start of one row of pixels and the start of the next row in memory. 

Hardware Alignment
~~~~~~~~~~~~~~~~~~
Stride optimizes memory access patterns and accelerates image processing pipelines. CPUs read data most efficiently when row memory addresses align with hardware boundaries, typically matching the CPU cache line size (e.g., multiples of 64 bytes, on a modern CPU architecture).

When the raw data width of an image row does not align with these boundaries, the system appends invisible **padding bytes** to the end of the row. Consequently, stride is often larger than the raw pixel data width.

Mathematical Formulas
~~~~~~~~~~~~~~~~~~~~~

**Minimum Data Width:**

.. code-block:: text

   data_width = width * channels * bytes_per_pixel

* **width**: Total number of columns in the image.
* **channels**: Total number of color channels (e.g., 3 for RGB).
* **bytes_per_pixel**: Data size allocated per pixel channel (e.g., 1 byte for 8-bit images).

**Actual Stride (With Padding):**

.. code-block:: text

   stride = round_up(data_width, alignment_boundary)

**Total Memory Footprint:**

.. code-block:: text

   size = height * stride

* **height**: Total number of rows in the image.

.. note::
   The structural order of color channels depends entirely on the framework hardware implementation. For example, OpenCV natively reads images in BGR (Blue, Green, Red) byte sequence rather than standard RGB.

Image Recognition Models
------------------------

Most modern image recognition architectures are built on **Convolutional Neural Networks (CNNs)**. Unlike standard fully connected networks, CNNs preserve the spatial relationships between pixels by applying local filters that extract structural patterns like edges, textures, and shapes. These filters are learned dynamically during training, enabling automatic feature extraction directly from raw pixel arrays.

CNNs learn the complex mapping between pixel arrangements and target object classes by training on large, labeled datasets. The final output layer typically utilizes a **Softmax activation function** to yield a probability distribution across all configured classes.

Tensor Layout Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before processing, raw images are structured into multi-dimensional arrays optimized for hardware acceleration:

.. code-block:: text

   Input Tensor Shape: [Batch Size, Height, Width, Channels]

* **Batch Size**: The number of images processed simultaneously during a single training step or inference pass.
* **Height & Width**: The spatial dimensions of the image canvas.
* **Channels**: The color depth components (e.g., 3 for RGB/BGR).

The Inference Pipeline
~~~~~~~~~~~~~~~~~~~~~~

Before executing a forward inference pass, input data must undergo a rigorous preprocessing pipeline to match the model's expected execution environment:

1. **Resizing**: Adjusts the variable dimensions of raw source images to match the fixed resolution of the network input layer.
2. **Normalization**: Scales raw pixel intensities from integer ranges (e.g., 0-255) to bounded floating-point distributions (such as ``[0, 1]`` or ``[-1, 1]``) to stabilize mathematical gradients.
3. **Type Conversion**: Casts the raw layout into a framework-compatible tensor format (e.g., FP32 or INT8).

Letterbox Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~

**Letterbox preprocessing** scales input images to a target fixed size while strictly preserving their original aspect ratio. 

Standard resizing techniques can stretch or squash an image, creating geometric distortions that degrade model classification accuracy. The letterbox pipeline avoids this distortion using a two-step approach:

* **Proportional Scaling**: Resizes the image so its longest dimension perfectly matches the boundaries of the target input canvas.
* **Canvas Padding**: Fills any remaining empty margins along the shorter dimension with a uniform filler color (typically black or neutral gray pixels) to meet the exact dimensions of the target tensor.

Visual Representation
^^^^^^^^^^^^^^^^^^^^^

The diagram below contrasts a standard destructive resize (which distorts the object geometry) against the **Letterbox pipeline** (which scales proportionally and appends uniform padding):

.. code-block:: text

   1. Raw Input Image (Wide Aspect Ratio)
   +---------------------------------------+

   |                  /\                   |  Height (H)
   |                /    \                 |  
   +---------------------------------------+
                    Width (W)

   2. Standard Direct Resize (Distorted / Squished Geometry)
   +-----------------------+

   |          /\           |  Target Height (TH)
   |        /    \         |  (Object shape is altered)
   +-----------------------+
       Target Width (TW)

   3. Letterbox Preprocessing (Preserved Geometry + Padding)
   +-----------------------+

   |:::::::::::::::::::::::| <--- Top Padding Margins
   |          /\           |  
   |        /    \         |  <--- Proportionally Scaled Image
   |:::::::::::::::::::::::| <--- Bottom Padding Margins
   +-----------------------+
       Target Width (TW)

Tensor Packing and Normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Following resizing and letterbox padding, the raw pixel array must be transformed into a contiguous tensor structure. This phase restructures memory layouts, shifts channel configurations, and scales numeric ranges to match the target execution engine.

Memory Layout and Channel Sequencing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Dimension Permutation (Format Layout)**:
  
  * Reorders data from interleaved layouts like **HWC** (Height, Width, Channels), standard in OpenCV, to planar layouts like **CHW** (Channels, Height, Width), required by frameworks like ONNX.
  
  * Ensures memory contiguity to maximize hardware cache hits during execution loops.

* **Channel Reordering (Color Space Alignment)**:
  
  * Swaps channel indices to match the training dataset topology.
  
  * Typically involves converting native camera pipelines from **BGR** format to standard deep learning **RGB** format (or vice versa).

Numeric Normalization
^^^^^^^^^^^^^^^^^^^^^

Raw integer intensities are converted into floating-point numbers to stabilize mathematical gradients and align with model parameters.

* **Min-Max Scaling**: Maps pixel arrays from native integer depths `` to fractional ranges `[0.0, 1.0]`.

* **Standardization (Z-Score)**: Shifts data using dataset-specific mean ($\mu$) and standard deviation ($\sigma$) parameters across each individual channel:

  .. code-block:: text

     normalized_pixel = (pixel - mean) / std

Model Architecture Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **YOLOv8 Optimization**: 
  YOLOv8 architectures (such as `yolov8n`) purposefully bypass Z-score standardization. They rely strictly on **Min-Max Scaling** (dividing by `255.0f`). The network utilizes internal batch normalization (`BatchNorm2d`) layers during execution, making external mean and standard deviation adjustments redundant.

* **Dataset Agnosticism**: 
  Skipping Z-score standardization avoids coupling the inference pipeline to rigid statistics from static training corpuses (like ImageNet). This ensures the model remains robust when deployed across diverse, dynamic edge environments (e.g., changing illumination profiles in outdoor industrial surveillance).

Hardware and Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Data Type Precision and Memory Inflation**:
  Converting an array from 8-bit unsigned integers (`uint8_t`) to 32-bit floating-point decimals (`float`) expands the memory footprint of the tensor by exactly **4x** (from 1 byte per channel pixel to 4 bytes). To conserve system memory, always perform this scaling as the absolute final step of the preprocessing pipeline.

* **NPU/TPU Quantization Constraints**:
  While modern deep learning frameworks handle input layers in standard FP32 formats, edge-native AI hardware accelerators often operate on quantized precision models. Depending on your runtime environment:
  
  * **FP16 / Half-Precision**: Truncates 32-bit float outputs down to 16 bits to maximize processing speed on modern embedded GPUs.
  
  * **INT8 Optimization**: Some specialized Edge NPUs bypass floating-point expansion entirely. They ingest the raw `uint8_t` pixels directly, executing the `[0, 255]` to `[0.0, 1.0]` conversion implicitly using mathematical integer scales and zero-point offsets embedded directly inside the compiled hardware binary.

