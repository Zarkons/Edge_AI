import pathlib
import tensorflow as tf
from typing import Callable, Iterable, List, Optional
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

class ConvertToTFLite:
    """A reusable class to convert any trained Keras/SavedModel to TFLite format 
    with full integer quantization using an externally provided data generator.
    """
    def __init__(self, model_path: str):
        """Initialize the converter by freezing the saved model variables.

        Args:
            model_path (str): Path to the directory containing the SavedModel.
        """
        loaded = tf.saved_model.load(str(model_path))
        serving_fn = loaded.signatures['serving_default']
        frozen_fn = convert_variables_to_constants_v2(serving_fn)

        self.loaded_model = loaded
        self.frozen_fn = frozen_fn
        self.converter = tf.lite.TFLiteConverter.from_concrete_functions([self.frozen_fn], self.loaded_model)
        self.tflite_model = None 

    def _create_converter(self) -> tf.lite.TFLiteConverter:
        """Create a fresh converter instance for each conversion attempt."""
        return tf.lite.TFLiteConverter.from_concrete_functions([self.frozen_fn], self.loaded_model)

    def convert_to_tflite(
        self,
        data_generator: Optional[Callable[[], Iterable[List[tf.Tensor]]]] = None,
        strict_int8_first: bool = False,
    ) -> bytes:
        """Convert the model to TFLite format with adaptive optimization scaling.

        Args:
            data_generator (Callable, optional): A generator function that yields 
                a list of tensors matching the input signature of the model.
            strict_int8_first (bool): Attempt strict hardware-only INT8 compilation.

        Returns:
            bytes: The serialized TFLite binary model data.
        """
        if strict_int8_first and data_generator is not None:
            self.converter = self._create_converter()
            self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
            self.converter.representative_dataset = data_generator
            self.converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            ]
            self.converter.target_spec.supported_types = [tf.int8]
            self.converter.inference_input_type = tf.int8
            self.converter.inference_output_type = tf.int8

            try:
                self.tflite_model = self.converter.convert()
                return self.tflite_model
            except Exception as exc:
                print(
                    "Strict INT8 conversion failed; retrying with Select TF ops "
                    "for unsupported control-flow/postprocessing ops."
                )
                print(f"Initial converter error: {exc}")

        # Fallback track for models with TensorArray/while/NMS-style ops.
        self.converter = self._create_converter()
        
        # Enable basic weight compression optimizations across the board
        self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        if data_generator is not None:
            self.converter.representative_dataset = data_generator
            
        self.converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        
        # Keep default float input/output dtypes for Flex compatibility.
        self.tflite_model = self.converter.convert()
        return self.tflite_model

    def save_tflite_model(self, file_path: str):
        """Save the generated TFLite binary bytes to a physical file."""
        print("Saving TFLite model to:", file_path)
        pathlib.Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            f.write(self.tflite_model)
