import os
import pathlib
import tensorflow as tf
import numpy as np
from absl import app
import ai_edge_litert.interpreter as tflite_interp

from learning.example.simple_audio_recognition.train import audio_processing
from learning.example.simple_audio_recognition.train import sample_processing

def run_inference(tflite_path: str, wav_path: str):
    # 1. Initialize the Interpreter
    interpreter = tflite_interp.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    # CRITICAL FIX: Extract the first layer dictionary from the lists directly
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # 2. Fetch raw audio wave
    float_audio_input = audio_processing.prepare_waveform_input(wav_path)
    
    # 3. Compute 4D spectrogram matrix using our helper -> shape [1, 124, 129, 1]
    spectrogram_data = audio_processing.get_spectrogram_4D(float_audio_input)
    spectrogram_data = tf.cast(spectrogram_data, tf.float32).numpy()
    
    # 4. Extract INT8 input quantization values (Now works perfectly with structural indexing)
    input_quant_info = input_details['quantization_parameters']
    input_scale = input_quant_info['scales']
    input_zero_point = input_quant_info['zero_points']
    
    # 5. Quantize the 4D spectrogram array to INT8
    int8_spectrogram = np.round(spectrogram_data / input_scale) + input_zero_point
    int8_spectrogram = np.clip(int8_spectrogram, -128, 127).astype(np.int8)
    
    # 6. Bind tensor array and run model
    interpreter.set_tensor(input_details['index'], int8_spectrogram)
    interpreter.invoke()
    
    # 7. Extract raw INT8 scores and de-quantize them back to floats
    raw_int8_outputs = interpreter.get_tensor(output_details['index'])
    output_quant_info = output_details['quantization_parameters']
    output_scale = output_quant_info['scales']
    output_zero_point = output_quant_info['zero_points']
    
    float_logits = (raw_int8_outputs.astype(np.float32) - output_zero_point) * output_scale
    
    # 8. Post-Processing: Softmax and Argmax
    exp_logits = np.exp(float_logits - np.max(float_logits, axis=-1, keepdims=True))
    probabilities = (exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)).flatten()
    
    predicted_id = np.argmax(probabilities)
    confidence = probabilities[predicted_id]
    label_names = sample_processing.get_label_names()
    predicted_word = label_names[predicted_id] if predicted_id < len(label_names) else f"ID_{predicted_id}"
    
    # print("\n" + "="*30)
    # print(f"🎯 Prediction:   {predicted_word.upper()}")
    # print(f"📊 Confidence:   {confidence * 100:.2f}%")
    # print("="*30)

    return predicted_word


def main(_):
    build_dir = sample_processing.get_build_dir()
    tflite_save_path = build_dir / "converted_audio_model/converted_model.tflite"

    for i in range(100):
        if i == 0:
            correct = 0
            incorrect = 0
        test_sample_wav = sample_processing.load_random_audio_sample(build_dir)

        predicted_word = run_inference(tflite_save_path, test_sample_wav)
        extracted_command = sample_processing.extract_command_from_filename(test_sample_wav)
        #print(f"📝 Ground Truth: {extracted_command.upper()}")

        if extracted_command.lower() == predicted_word.lower():
            correct+= 1
        else:
            incorrect+= 1

    print(f"✅ Correct: {correct} | ❌ Incorrect: {incorrect} | Total: {correct + incorrect}")
    print(f"Accuracy: {correct / (correct + incorrect) * 100:.2f}%")

if __name__ == "__main__":
    app.run(main)   