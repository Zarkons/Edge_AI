import argparse
import os
import sys
import tensorflow as tf
from python.runfiles import runfiles

def main():
    # 1. Parse the command-line argument passed from Bazel
    parser = argparse.ArgumentParser(description="Inspect TFLite model quantization metrics.")
    parser.add_argument(
        "--model_path", 
        required=True, 
        help="Workspace-relative path or runfiles token for the TFLite model."
    )
    args = parser.parse_args()

    # 2. Initialize Bazel Runfiles to safely find the file in the sandbox
    r = runfiles.Create()
    abs_model_path = r.Rlocation(args.model_path)

    # Fallback to direct path check if not running inside a strict runfiles environment
    if not abs_model_path or not os.path.exists(abs_model_path):
        abs_model_path = args.model_path

    if not os.path.exists(abs_model_path):
        print(f"Error: Could not locate model file at: {args.model_path}", file=sys.stderr)
        sys.exit(1)

    # 3. Load the model binary directly
    interpreter = tf.lite.Interpreter(model_path=abs_model_path)
    
    # Allocate tensors to ensure internal structures are fully populated
    interpreter.allocate_tensors()
    
    all_tensors = interpreter.get_tensor_details()

    int8_count = 0
    float32_count = 0
    other_count = 0
    
    # Clean tracking set for operations
    unique_ops = set()

    print(f"{'Tensor Name':<70} | {'Data Type':<10}")
    print("-" * 85)

    for tensor in all_tensors:
        name = tensor['name']
        dtype_str = str(tensor['dtype'].__name__)

        if "int8" in dtype_str:
            int8_count += 1
        elif "float32" in dtype_str:
            float32_count += 1
        else:
            other_count += 1

        short_name = name[-67:] if len(name) > 67 else name
        print(f"{short_name:<70} | {dtype_str:<10}")

    # Query the interpreter's true operation details directly if available
    try:
        ops_details = interpreter._get_ops_details()
        for op in ops_details:
            if 'op_name' in op:
                unique_ops.add(str(op['op_name']).upper())
    except AttributeError:
        # Fallback text parser if direct op extraction is blocked
        for tensor in all_tensors:
            name = tensor['name'].upper()
            
            # Split off standard namespace separators safely
            clean_name = name.split(':')[-1].split('/')[-1]
            if ';' in clean_name:
                clean_name = clean_name.split(';')[0]
                
            for prefix in ["TF.MATH.", "TF.NN.", "TF.COMPAT.V1.", "TF.IMAGE.", "TF."]:
                if clean_name.startswith(prefix):
                    clean_name = clean_name[len(prefix):]
            
            if "RESIZENEARESTNEIGHBOR" in clean_name or "RESIZE" in clean_name:
                clean_name = "RESIZE"
            elif "CONVOLUTION" in clean_name or "CONV_2D" in clean_name:
                clean_name = "CONVOLUTION"
                
            valid_ops = ['ADD', 'CONCAT', 'CONVOLUTION', 'MUL', 'PAD', 'RESHAPE', 'RESIZE', 'SPLIT', 'MULTIPLY']
            if any(op_type in clean_name for op_type in valid_ops):
                if clean_name == "MULTIPLY": 
                    clean_name = "MUL"
                clean_name = ''.join([i for i in clean_name if not i.isdigit()])
                unique_ops.add(clean_name)

    # Securely calculate file size strings
    try:
        file_bytes = os.path.getsize(abs_model_path)
        file_size_mb = file_bytes / (1024 * 1024)
        file_size_kb = file_bytes / 1024
        size_str = f"{file_size_mb:.2f} MB ({file_size_kb:.2f} KB)"
    except OSError:
        size_str = "Error reading file size"

    # Print the final unique operations breakdown cleanly
    print("\n" + "=" * 40)
    print("        UNIQUE OPERATIONS LIST")
    print("=" * 40)
    if unique_ops:
        for idx, op in enumerate(sorted(unique_ops), 1):
            print(f"  {idx}. {op}")
    else:
        print("  No explicit operational layers could be extracted.")
    print("-" * 40)

    print("\n" + "=" * 40)
    print("        QUANTIZATION METRICS")
    print("=" * 40)
    total = int8_count + float32_count + other_count
    print(f"Total Tensors:          {total}")
    print(f"Quantized INT8 Nodes:   {int8_count} ({ (int8_count/total)*100 :.1f}%)")
    print(f"Floating-Point Float32: {float32_count} ({ (float32_count/total)*100 :.1f}%)")
    print(f"Other Nodes (e.g. i32): {other_count}")
    print("-" * 40)
    print(f"TFLite Model File Size: {size_str}")
    print("=" * 40)

if __name__ == '__main__':
    main()
