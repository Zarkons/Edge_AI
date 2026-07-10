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

.. mermaid:: diagrams/model.mmd
   :align: center
   :caption: Structural CNN Inference Topology Matrix


Conversion to TFLite
--------------------

.. autoclass:: convert_to_tflite.ConvertToTFLite
    :members: