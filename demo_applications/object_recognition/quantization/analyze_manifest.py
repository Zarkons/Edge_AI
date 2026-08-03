import json
import os
import pathlib
import sys
from collections import Counter

def analyze_manifest_extended(json_path):
    if not os.path.exists(json_path):
        print(f"Error: Target file not found at: {json_path}")
        return

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON syntax formatting inside manifest: {e}")
        return

    pipeline = data.get("optimized_execution_pipeline") or data.get(
        "execution_graph", data
    )
    if not isinstance(pipeline, list):
        pipeline = [pipeline]

    excluded_ops = {
        "Split",
        "Slice",
        "Reshape",
        "Transpose",
        "DequantizeLinear",
        "QuantizeLinear",
    }

    global_op_counts = Counter()
    global_attr_counts = Counter()
    
    # Track a set of all unique attribute keys strictly from non-excluded nodes
    non_excluded_attribute_keys = set()
    filtered_op_attributes = {}

    print(f"Processing {len(pipeline)} nodes from manifest execution graph...\n")

    for node in pipeline:
        if not isinstance(node, dict):
            continue

        op_type = node.get("op_type", "UNKNOWN_OP")
        global_op_counts[op_type] += 1

        attributes_block = node.get("attributes", {})
        node_attrs = {}

        if isinstance(attributes_block, dict):
            node_attrs = attributes_block
        elif isinstance(attributes_block, list):
            for item in attributes_block:
                if isinstance(item, dict):
                    node_attrs.update(item)

        for attr_key in node_attrs.keys():
            global_attr_counts[attr_key] += 1

        # Check if the node is within our active computation path
        if op_type in excluded_ops:
            continue

        # Add to our consolidated list of computational attributes
        for attr_key in node_attrs.keys():
            non_excluded_attribute_keys.add(attr_key)

        if op_type not in filtered_op_attributes:
            filtered_op_attributes[op_type] = {}

        for attr_key, attr_val in node_attrs.items():
            if attr_key not in filtered_op_attributes[op_type]:
                filtered_op_attributes[op_type][attr_key] = {
                    "count": 0,
                    "sample": attr_val,
                }
            filtered_op_attributes[op_type][attr_key]["count"] += 1

    print("=" * 60)
    print(f"{'GLOBAL OPERATIONAL TYPE PROFILE (op_type)':<45}{'COUNT':<15}")
    print("=" * 60)
    for op, count in global_op_counts.most_common():
        marker = " [EXCLUDED FROM DEEP SCAN]" if op in excluded_ops else ""
        print(f"{op + marker:<45}{count:<15}")

    print("\n" + "=" * 60)
    print("DEEP ANALYSIS: ATTRIBUTES INSIDE COMPUTATIONAL TRACK")
    print("(Excluding: Split, Slice, Reshape, Transpose, QDQ Layers)")
    print("=" * 60)

    # NEW SECTION: Flat list summary of all unique keys from non-excluded nodes
    print("★ ALL UNIQUE ATTRIBUTE KEYS REQUIRING C++ STRUCT FIELD MAPPING:")
    if not non_excluded_attribute_keys:
        print("  [None Found]")
    else:
        for attr_key in sorted(non_excluded_attribute_keys):
            print(f"  - {attr_key}")
    print("-" * 60)

    if not filtered_op_attributes:
        print("No active heavy math computation layers found matching filter.")
    else:
        for op_type, attr_data in sorted(filtered_op_attributes.items()):
            print(f"\n▶ OPERATOR: [ {op_type} ]")
            print(f"  {'Attribute Key':<25}{'Frequency':<15}{'Data Sample / Format':<20}")
            print("  " + "-" * 58)
            for attr_key, metrics in sorted(attr_data.items()):
                sample_str = str(metrics["sample"])
                if len(sample_str) > 25:
                    sample_str = sample_str[:22] + "..."

                print(f"  {attr_key:<25}{metrics['count']:<15}{sample_str:<20}")
    print("=" * 60)


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
    target_manifest = "yolov8n_manifest.json"
    optimized_manifest = "optimized_execution_graph.json"

    if len(sys.argv) > 1:
        target_manifest = sys.argv[1]

    analyze_manifest_extended(output_dir / target_manifest)
