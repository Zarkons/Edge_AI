import os
import pathlib
import tensorflow as tf
import numpy as np
from absl import app
import ai_edge_litert.interpreter as tflite_interp

LABEL_NAMES = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]

def get_build_dir():
    workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    
    if workspace_dir:
        actual_save_dir = os.path.join(workspace_dir, "learning/example/simple_audio_recognition/train/build_dir/")
    else:
        # 2. Universal Fallback: Calculate path dynamically relative to this file's position
        # (This handles standard python runs, VS Code debugging, or bazel tests)
        current_file_path = os.path.abspath(__file__)
        # Adjust the number of parents based on your folder depth to reach the root
        repo_root = pathlib.Path(current_file_path).parents[2] 
        actual_save_dir = os.path.join(repo_root, "learning/example/simple_audio_recognition/train/build_dir/")
        
    # Ensure the directory actually exists on your Mac before saving files to it
    os.makedirs(actual_save_dir, exist_ok=True)
    return pathlib.Path(actual_save_dir)

def load_and_preprocess_audio(wav_path: str) -> np.ndarray:
    """Read an audio file, normalize it, and pad/truncate to exactly 16000 samples natively in TF."""
    # Read the binary file directly from disk using native TF
    audio_binary = tf.io.read_file(str(wav_path))
    
    # Decode wav directly (this handles float32 normalization between -1.0 and 1.0 automatically)
    audio, _ = tf.audio.decode_wav(audio_binary, desired_channels=1, desired_samples=16000)
    audio = tf.squeeze(audio, axis=-1) # Drop channel dimension -> shape 
    
    # Explicit padding guarantee (in case the file was shorter than 16000)
    audio = audio[:16000]
    if tf.shape(audio)[0] < 16000:
        zero_padding = tf.zeros([16000] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], axis=0)
        
    # Convert back to numpy array and add batch dimension -> shape [1, 16000]
    return audio.numpy()[np.newaxis, :]

def get_spectrogram(waveform):
    """
    Converts a 1D/2D float32 audio waveform into a 4D spectrogram matrix.
    Input shape: [1, 16000] or [16000]
    Output shape: [1, 124, 129, 1] (Fully matching Keras Conv2D expectations)
    """
    # 1. Ensure the waveform is floating-point data
    waveform = tf.cast(waveform, tf.float32)
    
    # 2. If it doesn't have a batch dimension, flatten it and squeeze to 1D
    # tf.signal.stft natively expects a 1D vector or [Batch, Samples]
    waveform = tf.squeeze(waveform) 
    
    # 3. Compute the Short-Time Fourier Transform (STFT)
    # These standard hyper-parameters yield a 124 x 129 matrix for 16,000 samples
    stft = tf.signal.stft(
        waveform, 
        frame_length=255,   # Window size
        frame_step=128      # Hop size (overlap)
    )
    
    # 4. Extract the absolute magnitude (drops phase information, leaving frequencies)
    spectrogram = tf.abs(stft)
    
    # 5. Add the mandatory Batch dimension and Channel dimension to make it 4D
    # [124, 129] -> [1, 124, 129, 1]
    spectrogram = tf.expand_dims(spectrogram, axis=0)  # Add Batch dimension at front
    spectrogram = tf.expand_dims(spectrogram, axis=-1) # Add Channel dimension at back
    
    return spectrogram

def run_inference(tflite_path: str, wav_path: str):
    # 1. Initialize the Interpreter
    interpreter = tflite_interp.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    # CRITICAL FIX: Extract the first layer dictionary from the lists directly
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # 2. Fetch raw audio wave
    float_audio_input = load_and_preprocess_audio(wav_path)
    
    # 3. Compute 4D spectrogram matrix using our helper -> shape [1, 124, 129, 1]
    spectrogram_data = get_spectrogram(float_audio_input)
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
    print(f"Running true INT8 inference on: {pathlib.Path(wav_path).name}...")
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
    predicted_word = LABEL_NAMES[predicted_id] if predicted_id < len(LABEL_NAMES) else f"ID_{predicted_id}"
    
    print("\n" + "="*30)
    print(f"🎯 Prediction:   {predicted_word.upper()}")
    print(f"📊 Confidence:   {confidence * 100:.2f}%")
    print("="*30)


def main(_):
    build_dir = get_build_dir()
    tflite_save_path = build_dir / "converted_audio_model/converted_model.tflite"
    test_sample_wav = build_dir / "data/mini_speech_commands_extracted/mini_speech_commands/no/0ab3b47d_nohash_0.wav"
    run_inference(tflite_save_path, test_sample_wav)

if __name__ == "__main__":
    app.run(main)   