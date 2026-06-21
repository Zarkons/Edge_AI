import os
import pathlib
import tensorflow as tf
from absl import app
import random
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from learning.example.simple_audio_recognition.train import audio_processing
from learning.example.simple_audio_recognition.train import sample_processing

random.seed(42)
LABEL_NAMES = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]


class ConvertToTFLite:
    def __init__(self, model_path: str):
        loaded = tf.saved_model.load(str(model_path))
        serving_fn = loaded.signatures['serving_default']
        frozen_fn = convert_variables_to_constants_v2(serving_fn)

        self.converter = tf.lite.TFLiteConverter.from_concrete_functions([frozen_fn])
        self.tflite_model = None  # Placeholder for the converted TFLite model

    def representative_data_gen(self, inputs_path: str):
        audio_dir = pathlib.Path(inputs_path)
        wav_paths = list(audio_dir.glob("**/*.wav"))

        random.shuffle(wav_paths)
        wav_paths = wav_paths[:100]

        for wav_path in wav_paths:
            try:
                audio_bytes = tf.io.read_file(str(wav_path))
                audio, _ = tf.audio.decode_wav(
                    audio_bytes,
                    desired_channels=1,
                    desired_samples=16000,
                )
                audio = tf.squeeze(audio, axis=-1)
                spectrogram = audio_processing.get_spectrogram_3D(audio)
                spectrogram = spectrogram[tf.newaxis, ...]

                # For a single-input signature, representative samples should be yielded
                # as a positional list with one tensor.
                yield [tf.cast(spectrogram, tf.float32)]

            except Exception as e:
                # Skip any corrupt audio files safely
                continue


    def convert_to_tflite(self, inputs_path: str) -> bytes:
        """Convert the Keras model to TFLite format."""

        # Enable full integer quantization for Edge AI hardware optimization
        self.converter.optimizations = [tf.lite.Optimize.DEFAULT]
        self.converter.representative_dataset = lambda: self.representative_data_gen(inputs_path)

        # Enforce integer execution if your edge hardware requires it
        self.converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        ]
        self.converter.target_spec.supported_types = [tf.int8]

        self.converter.inference_input_type = tf.int8
        self.converter.inference_output_type = tf.int8

        self.tflite_model = self.converter.convert()
        return self.tflite_model

    def save_tflite_model(self, file_path: str):
        """Save the generated TFLite binary bytes to a file."""
        print("Saving TFLite model to:", file_path)
        pathlib.Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            f.write(self.tflite_model) # Pass and write the actual converted bytes

def main(_):
    build_dir = sample_processing.get_build_dir()
    model_path = build_dir / "exported_trained_keras_model"
    converter = ConvertToTFLite(model_path)
    inputs_path = build_dir / "data/mini_speech_commands_extracted/mini_speech_commands"
    tflite_model = converter.convert_to_tflite(inputs_path)
    tflite_save_path = build_dir / "converted_audio_model/converted_model.tflite"
    converter.save_tflite_model(tflite_save_path)


if __name__ == "__main__":
    app.run(main)