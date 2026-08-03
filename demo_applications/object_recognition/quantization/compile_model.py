import onnx
from onnx import numpy_helper
import json
import os
import pathlib

def main():
    # 1. RESOLVE ABSOLUTE PROJECT PATH TO ESCAPE BAZEL SANDBOX
    # If executed via 'bazel run', this env var points to your real project folder
    workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    
    if workspace_dir:
        # Base the output root directly on your real persistent disk folder
        project_root = pathlib.Path(workspace_dir)
    else:
        # Fallback if running as an isolated system test rule
        project_root = pathlib.Path(os.getcwd())
        
    # Build absolute paths pointing straight to your real project tree
    output_dir = project_root / "demo_applications/object_recognition/quantization/_build_dir/compiled_engine_assets"
    model_path = project_root / "demo_applications/object_recognition/quantization/_build_dir/models/yolov8n_mixed_precision.onnx"
    
    bin_out_path = output_dir / "yolov8n_weights.bin"
    json_out_path = output_dir / "yolov8n_manifest.json"
    
    # Ensure the entire nested directory tree exists on your real physical drive
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Could not locate the mixed-precision ONNX file at absolute path: {model_path}\n"
                                "Make sure you ran your quantization stage successfully first!")

    print(f"Loading hybrid ONNX graph from: {model_path}")
    model = onnx.load(str(model_path))
    graph = model.graph
    
    manifest = {
        "model_metadata": {
            "architecture": "YOLOv8n-Mixed-Precision",
            "input_tensor_name": "images",
            "input_shape": [1, 3, 640, 640]
        },
        "weight_tensors": {},
        "execution_graph": []
    }
    
    current_byte_offset = 0
    
    # --- PHASE 1: COMPRESS & FLATTEN ALL LEARNED WEIGHTS ---
    print("Phase 1: Packing tensors into contiguous flat binary space...")
    with open(bin_out_path, "wb") as bin_file:
        for initializer in graph.initializer:
            clean_name = initializer.name.replace("/", "_").replace(".", "_")
            numpy_array = numpy_helper.to_array(initializer)
            raw_bytes = numpy_array.tobytes()
            byte_length = len(raw_bytes)
            
            manifest["weight_tensors"][initializer.name] = {
                "cpp_safe_name": clean_name,
                "shape": list(numpy_array.shape),
                "data_type": str(numpy_array.dtype),
                "byte_offset": current_byte_offset,
                "byte_length": byte_length
            }
            
            bin_file.write(raw_bytes)
            current_byte_offset += byte_length

    # --- PHASE 2: TRACE THE TOPOLOGICAL INSTRUCTION MANIFEST ---
    print("Phase 2: Parsing operational nodes to build execution sequence...")
    for idx, node in enumerate(graph.node):
        node_entry = {
            "execution_order": idx,
            "op_type": node.op_type,
            "node_name": node.name if node.name else f"node_{idx}",
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attributes": {}
        }
        
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.INTS:
                node_entry["attributes"][attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.INT:
                node_entry["attributes"][attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.FLOAT:
                node_entry["attributes"][attr.name] = attr.f
                
        manifest["execution_graph"].append(node_entry)

    # --- PHASE 3: WRITE OUT TOPOLOGICAL MANIFEST ---
    with open(json_out_path, "w") as json_file:
        json.dump(manifest, json_file, indent=2)
        
    print("\n-------------------------------------------------------------------")
    print(f"🟩 COMPILATION SUCCESS!")
    print(f"📦 Contiguous Weight Flat Buffer: {bin_out_path}")
    print(f"   ↳ Total Packed Buffer Size:   {bin_out_path.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"📄 Topological Operation Map:   {json_out_path}")
    print("-------------------------------------------------------------------\n")

if __name__ == "__main__":
    main()
