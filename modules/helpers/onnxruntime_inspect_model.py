import argparse
import os
import sys
import onnx
from python.runfiles import runfiles

def main():
    # 1. Parse the command-line argument passed from Bazel
    parser = argparse.ArgumentParser(description="Inspect ONNX model quantization metrics.")
    parser.add_argument(
        "--model_path", 
        required=True, 
        help="Workspace-relative path or runfiles token for the ONNX model."
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

    print(f"Inspecting ONNX Model Target: {abs_model_path}\n")

    # 3. Load the ONNX model graph structure
    try:
        model = onnx.load(abs_model_path)
    except Exception as e:
        print(f"Error: Failed to parse ONNX protobuf data: {e}", file=sys.stderr)
        sys.exit(1)
        
    graph = model.graph

    # Complete mapping of standard ONNX TensorProto enum integers to readable strings
    type_map = {
        1: "FLOAT32",
        2: "UINT8",
        3: "INT8",
        4: "UINT16",
        5: "INT16",
        6: "INT32",
        7: "INT64",
        9: "BOOL",
        10: "FLOAT16",
        11: "DOUBLE"
    }

    # Initialize counter variables matching your exact original format rules
    int8_count = 0
    float32_count = 0
    uint8_count = 0
    int32_count = 0
    int64_count = 0
    other_count = 0
    
    unique_ops = set()

    print(f"{'Weight/Initializer Name':<70} | {'Data Type':<10}")
    print("-" * 85)

    for initializer in graph.initializer:
        dtype_int = initializer.data_type
        dtype_str = type_map.get(dtype_int, f"UNKNOWN({dtype_int})")

        # Explicitly check and route every known tensor data type
        if dtype_int == 3:      # INT8
            int8_count += 1
        elif dtype_int == 1:    # FLOAT32
            float32_count += 1
        elif dtype_int == 2:    # UINT8
            uint8_count += 1
        elif dtype_int == 6:    # INT32
            int32_count += 1
        elif dtype_int == 7:    # INT64
            int64_count += 1
        else:
            other_count += 1

        short_name = initializer.name[-67:] if len(initializer.name) > 67 else initializer.name
        print(f"{short_name:<70} | {dtype_str:<10}")

    for node in graph.node:
        unique_ops.add(node.op_type)

    # Securely calculate file size strings
    try:
        file_bytes = os.path.getsize(abs_model_path)
        size_str = f"{file_bytes / (1024 * 1024):.2f} MB"
    except OSError:
        size_str = "Unknown"

    # Print the final unique operations breakdown cleanly
    print("\n" + "=" * 40)
    print("        UNIQUE OPERATIONS LIST (OPS)")
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
    total = int8_count + float32_count + uint8_count + int32_count + int64_count + other_count
    
    print(f"Total Tensors:          {total}")
    print(f"Quantized INT8 Nodes:   {int8_count} ({ (int8_count/total)*100 if total > 0 else 0 :.1f}%)")
    print(f"Floating-Point Float32: {float32_count} ({ (float32_count/total)*100 if total > 0 else 0 :.1f}%)")
    print(f"Quantized UINT8 Nodes:  {uint8_count} ({ (uint8_count/total)*100 if total > 0 else 0 :.1f}%)")
    print(f"Integer INT32 Nodes:    {int32_count} ({ (int32_count/total)*100 if total > 0 else 0 :.1f}%)")
    print(f"Integer INT64 Nodes:    {int64_count} ({ (int64_count/total)*100 if total > 0 else 0 :.1f}%)")
    print(f"Other Nodes (e.g. f16): {other_count}")
    print("-" * 40)
    
    if "QuantizeLinear" in unique_ops or "DequantizeLinear" in unique_ops:
        print("Quantization Status:   🟩 TRUE STATIC INT8")
    else:
        print("Quantization Status:   ⚠️ UNQUANTIZED Baseline")
        
    print(f"Model File Size:        {size_str}")
    print("=" * 40)

if __name__ == '__main__':
    main()
