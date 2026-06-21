from absl import app

import numpy as np
import tensorflow as tf
import keras

from learning.example.simple_audio_recognition.train import audio_processing
from learning.example.simple_audio_recognition.train import sample_processing



def run_inference(model, waveform):
    """Run inference on a single waveform input."""
    spectrogram = audio_processing.get_spectrogram_4D(waveform)

    outputs_dict = model(spectrogram, training=False)
    first_key = list(outputs_dict.keys())[0]
    logits = outputs_dict[first_key]

    probabilities = tf.nn.softmax(logits)

    return probabilities

def main(_):
    build_dir = sample_processing.get_build_dir()
    exported_model_path = build_dir / "exported_trained_keras_model"

    tf_sm_layer = keras.layers.TFSMLayer(
        exported_model_path,
        call_endpoint="serving_default"
    )

    model = tf.keras.Sequential([tf_sm_layer])

    for i in range(100):
        if i == 0:
            correct = 0
            incorrect = 0

        # Load a random audio sample from the dataset
        test_sample_wav = sample_processing.load_random_audio_sample(build_dir)

        # Prepare the waveform input
        float_audio_input = audio_processing.prepare_waveform_input(test_sample_wav)

        # Run inference
        probabilities = run_inference(model, float_audio_input)

        predicted_id = np.argmax(probabilities)
        confidence = probabilities[0][predicted_id].numpy()
        label_names = sample_processing.get_label_names()
        predicted_word = label_names[predicted_id] if predicted_id < len(label_names) else f"ID_{predicted_id}"

        extracted_command = sample_processing.extract_command_from_filename(test_sample_wav)

        if extracted_command.lower() == predicted_word.lower():
            correct += 1
        else:
            incorrect += 1

        # print(f"Test Sample {i+1}:")
        # print(f" - Ground Truth: {extracted_command.upper()}")
        # print(f" - Predicted:    {predicted_word.upper()} (Confidence: {confidence * 100:.2f}%)")
    print(f" - Current Accuracy: {correct / (correct + incorrect) * 100:.2f}%\n")


if __name__ == "__main__":
    app.run(main)