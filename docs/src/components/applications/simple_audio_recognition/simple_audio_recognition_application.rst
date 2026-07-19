Application
===========

This section covers the process of deploying and running a simple audio recognition model using the Edge AI framework. It includes steps for loading the trained model, preparing input audio data, and performing inference to obtain predictions.

Sample Processing
-----------------

File Input
^^^^^^^^^^
The application expects audio files in WAV format as input. Ensure that the audio files are properly formatted and contain the expected audio data.

.. doxygenfunction:: file_sample_processing::GetSample

Microphone Input
^^^^^^^^^^^^^^^^
The application can also capture audio input directly from a microphone. Ensure that the microphone is properly configured and accessible by the application.

.. doxygenclass:: mic_sample_processing::MicrophoneInput
    :members:

Prepare Samples for Inference
------------------------------
Before performing inference, the audio samples need to be preprocessed and converted into a suitable format for the model. This includes normalizing the audio data, extracting relevant features, and ensuring that the input dimensions match the model's expected input shape.

.. doxygennamespace:: audio_model_pipeline
    :members: