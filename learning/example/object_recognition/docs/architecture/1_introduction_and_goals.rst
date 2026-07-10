1. Introduction and Goals
=========================

This document describes the architecture of the Object Recognition example application. It is intended to provide a high-level overview of the system's design, its goals, and the rationale behind key architectural decisions.

1.1 Requirements Overview
-------------------------

.. req:: Object Recognition
    :id: req_object_recognition
    :status: draft
    :req_type: functional
    
        The system shall recognize and classify objects in real-time.

.. req:: Input Processing
    :id: req_input_processing
    :status: draft
    :req_type: functional
    
        The system shall be able to process inputs from live camera feeds as well as pre-recorded video files.

.. req:: Output Generation
    :id: req_output_generation
    :status: draft
    :req_type: functional

        The system shall output the recognized objects along with their confidence scores and bounding boxes to a user interface.

.. note::
    Usually the requirements are listed in a separate document, and referenced here, but for the sake of this example, they are included here to provide context for the architectural decisions.



1.2 Quality Goals
-----------------
The following overarching quality goals serve as the primary architectural drivers for this system. They represent the core qualities valued by our stakeholders:

.. goal:: High Inference Accuracy
  :id: goal_high_inference_accuracy
  :status: draft
  :characteristic: Accuracy

    The system shall achieve high accuracy in object recognition to minimize
    false negatives and ensure reliable performance in production environments.

.. goal:: Low Memory Footprint
  :id: goal_low_memory_footprint
  :status: draft
  :characteristic: Efficiency

    The system shall be optimized for low memory usage to enable deployment on resource-constrained edge devices.

.. goal:: Framework & Model Agnosticism
  :id: goal_framework_model_agnosticism
  :status: draft
  :characteristic: Maintainability

    The architecture shall be designed to allow for easy swapping of machine learning models and frameworks without requiring changes to the application code.

.. goal:: Runtime Extensibility
  :id: goal_runtime_extensibility
  :status: draft
  :characteristic: Flexibility

    The system shall be designed to support the integration of new model runtime frameworks (e.g., ONNX, TensorRT, OpenVINO) to enable deployment on various edge hardware platforms.

.. goal:: System Observability
  :id: goal_system_observability
  :status: draft
  :characteristic: Observability

    The system shall provide observability features for logging and performance monitoring to facilitate rapid diagnosis of issues and performance bottlenecks.

1.3 Stakeholders
----------------

.. note::
    No stakeholders have been identified for this example application. In a real-world scenario, stakeholders would include end-users, developers, and business owners who have an interest in the system's performance and functionality.
