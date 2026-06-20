import pathlib
import tensorflow as tf
import os

from learning.example.simple_audio_recognition.train import sample_processing

def inspect_tflite_datatypes(tflite_path: str):
    # 1. Load the converted binary model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    # 2. Extract details for every internal tensor (weights, biases, activations)
    tensor_details = interpreter.get_tensor_details()
    
    print(f"=== TFLite Tensor Inventory for: {pathlib.Path(tflite_path).name} ===")
    print(f"Total internal tensors found: {len(tensor_details)}\n")
    
    # Track counts of different data types
    dtype_counts = {}
    int8_layer_names = []
    
    for tensor in tensor_details:
        # Get data type (e.g., tf.int8, tf.float32, tf.int32)
        dtype = tensor['dtype']
        dtype_name = dtype.__name__ if hasattr(dtype, '__name__') else str(dtype)
        
        # Track counts
        dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
        
        # Log the internal Dense or Conv weights if they are INT8
        if dtype == tf.int8 and ('weights' in tensor['name'] or 'kernel' in tensor['name'] or 'dense' in tensor['name'].lower()):
            int8_layer_names.append(f" - {tensor['name']} (Shape: {tensor['shape']})")

    # 3. Print Data Type Breakdown
    print("--- Data Type Breakdown ---")
    for dtype, count in dtype_counts.items():
        print(f"  * {dtype}: {count} tensors")
    print("-" * 27 + "\n")
    
    # 4. Confirm weight status
    if int8_layer_names:
        print("✅ SUCCESS: Found quantized INT8 neural network parameters:")
        for layer in int8_layer_names[:10]:  # Limit print to first 10 for readability
            print(layer)
        if len(int8_layer_names) > 10:
            print(f" ... and {len(int8_layer_names) - 10} more INT8 layers.")
    else:
        print("❌ WARNING: No internal INT8 neural network layers found. The model might still be using float32.")

if __name__ == "__main__":
    build_dir = sample_processing.get_build_dir()
    tflite_model_path = build_dir / "converted_audio_model/converted_model.tflite"
    
    inspect_tflite_datatypes(tflite_model_path)
