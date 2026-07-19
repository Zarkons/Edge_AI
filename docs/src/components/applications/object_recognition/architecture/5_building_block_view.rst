5. Building Block View
======================

.. component:: Object Recognition System
   :id: comp_object_recognition_system
   :status: draft

.. mermaid:: diagrams/white_box_L1_system_view.mmd
   :align: center
   :caption: White Box View of the Object Recognition System


Component Decomposition
-----------------------

.. component:: Input Provider Factory
   :id: comp_input_provider_factory
   :status: draft
   :decomposed_from: comp_object_recognition_system

   Component responsible for creating input providers based on the input source type. It abstracts the details of how inputs are obtained, allowing the system to support various input sources seamlessly.

.. component:: Inference Engine
   :id: comp_inference_engine
   :status: draft
   :decomposed_from: comp_object_recognition_system

   Component responsible for performing inference on the input data using the loaded machine learning model and dedicated runtime framework for the target hardware.

.. component:: User Interface Provider
   :id: comp_user_interface_provider
   :status: draft
   :decomposed_from: comp_object_recognition_system

   Component responsible for providing a user interface for interacting with the object recognition system. It handles the ouput of the inference results and displays the infromation to the user in graphical or textual format.

.. component:: Observability Agent
    :id: comp_observability_agent
    :status: draft
    :decomposed_from: comp_object_recognition_system
    
    Component responsible for collecting and reporting metrics, logs, and traces from the object recognition system. It enables monitoring and observability of the system's performance and behavior.

.. component:: Object Recognition Main Application 
   :id: comp_object_recognition_main_app
   :status: draft
   :decomposed_from: comp_object_recognition_system

   Component responsible for coordinating the overall operation of the object recognition system. It integrates the various components, manages the workflow, and ensures that the system functions as a cohesive unit.
