import json
import os
import pathlib
import sys
from collections import Counter

def verify_pipeline_integrity(json_data):
    pipeline = json_data.get("optimized_execution_pipeline", [])
    
    # 1. Track the active registry of tensors generated in memory
    produced_tensors = {"images"} # Set baseline input tracking
    errors_found = 0
    
    print("--- STARTING TOPOLOGY INTEGRITY AUDIT ---")
    for step in pipeline:
        order = step.get("execution_order")
        op_type = step.get("op_type")
        node_name = step.get("node_name")
        inputs = step.get("inputs", [])
        outputs = step.get("outputs", [])
        
        # Validation Rule A: Input Dependency Presence Check
        for inp in inputs:
            # If the input name ends with a constant suffix, it's a weight, skip validation
            if "weight" in inp or "bias" in inp or "constant" in inp or "Constant" in inp:
                continue
            if inp not in produced_tensors:
                print(f"[ERROR][Order {order}] Node '{node_name}' ({op_type}) expects input '{inp}' which was never generated!")
                errors_found += 1
                
        # Register new generated feature maps
        for out in outputs:
            produced_tensors.add(out)
            
        # Validation Rule B: Layer Dimension Metric Sanity Checked
        if op_type == "Conv":
            attrs = step.get("attributes", {})
            if not attrs.get("strides") or not attrs.get("kernel_shape"):
                print(f"[ERROR][Order {order}] Conv Node '{node_name}' has empty configuration attributes!")
                errors_found += 1
                
        # Validation Rule C: Quantization Metric Safety Check
        q_params = step.get("quantization_parameters", {})
        if q_params.get("input_scale") == 1.0 and q_params.get("output_scale") == 1.0 and op_type == "Conv":
            print(f"[WARNING][Order {order}] Conv Node '{node_name}' uses unoptimized fallback scales (1.0). Verify lookups.")

    print(f"--- INTEGRITY AUDIT COMPLETE: {errors_found} CRITICAL ERRORS DETECTED ---\n")
    return errors_found == 0

if __name__ == "__main__":
    # Expects the file path as the first argument, or defaults to pipeline output

    workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    
    if workspace_dir:
        # Base the output root directly on your real persistent disk folder
        project_root = pathlib.Path(workspace_dir)
    else:
        # Fallback if running as an isolated system test rule
        project_root = pathlib.Path(os.getcwd())
        
    # Build absolute paths pointing straight to your real project tree
    output_dir = project_root / "demo_applications/object_recognition/quantization/_build_dir/compiled_engine_assets"
    optimized_manifest = "optimized_execution_graph.json"

    manifest_path = output_dir / optimized_manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # Pass the actual parsed dictionary data, not the path object!
    verify_pipeline_integrity(json_data)
