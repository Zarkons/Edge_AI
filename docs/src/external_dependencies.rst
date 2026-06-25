External Dependencies
=====================

Following external dependencies are needed as prerequisites for building and running the Edge AI project:
	- tflite-micro: TensorFlow Lite Micro is a lightweight version of TensorFlow Lite designed for microcontrollers and embedded devices. It provides a set of tools and libraries for running machine learning models on resource-constrained devices. GitHub repository:
		- https://github.com/tensorflow/tflite-micro.git
		It is included as a submodule in the Edge AI project, and it is automatically downloaded and built during the build process.
	- tensorflow-core: TensorFlow Core is the main library of TensorFlow, an open-source machine learning framework. It provides a comprehensive set of tools and APIs for building, training and deploying machine learning models. GitHub repository: 
		- https://github.com/tensorflow/tensorflow.git
		It is included as a submodule in the Edge AI project, and it is automatically downloaded and built during the build process.
	- miniaudio: Miniaudio is a single file audio playback and capture library. It is taken from the following repository:
		- https://github.com/mackron/miniaudio.git
		It is copied into the Edge AI project, and it is automatically built during the build process. For more information about miniaudio, please refer to the official documentation at:
		- https://miniaud.io/