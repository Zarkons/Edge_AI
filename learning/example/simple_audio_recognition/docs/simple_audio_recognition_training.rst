Training
========

This section covers the process of training a simple audio recognition model using the Edge AI framework. It includes steps for preparing the dataset, defining the model architecture, training the model, and evaluating its performance.

Model Creation
--------------

Following method is used to create the model:

.. autofunction:: train.get_model
    :noindex:


Model Architecture
------------------

.. image:: diagrams/audio_recognition_tflite_quantized.png
    :align: center
    :alt: Model Architecture


Conversion to TFLite
--------------------

.. autoclass:: convert_to_tflite.ConvertToTFLite
    :members: