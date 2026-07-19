import os
import pathlib
import sys
import shutil # 🟩 Added for file copying
import glob
import tensorflow as tf
from absl import app, flags
from ultralytics import YOLO
from python.runfiles import runfiles
from tensorflow.core.protobuf import saved_model_pb2

FLAGS = flags.FLAGS
flags.DEFINE_string('base_model', None, 'Runfiles lookup key for the base .pt file.')
flags.DEFINE_string('calibration_images', None, 'Runfiles space-separated list of image files.')

import os
import tensorflow as tf

def analyze_pb_model(pb_file_path):
    if not os.path.exists(pb_file_path):
        print(f"Error: File not found at {pb_file_path}")
        return

    # 1. Open and parse the file as a SavedModel protobuf
    with tf.io.gfile.GFile(pb_file_path, "rb") as f:
        saved_model = saved_model_pb2.SavedModel()
        saved_model.ParseFromString(f.read())

    if not saved_model.meta_graphs:
        print("Error: SavedModel does not contain any MetaGraphs.")
        return
        
    # 🟩 FIX: Extract the first MetaGraphDef out of the repeated list array
    meta_graph = saved_model.meta_graphs[0]
    
    unique_ops = set()
    total_tensors = 0
    int8_count = 0
    float32_count = 0
    other_count = 0

    # 2. Extract nested layers from the function library
    functions = meta_graph.graph_def.library.function
    
    if functions:
        print("Unpacking nested TensorFlow v2 function library layers...")
        for function in functions:
            for node in function.node_def:
                unique_ops.add(node.op)
                
                # Check the exact payload type inside Constant nodes
                if node.op == "Const" and "value" in node.attr:
                    total_tensors += 1
                    dtype_val = node.attr["value"].tensor.dtype
                    if dtype_val in [tf.int8.as_datatype_enum, tf.qint8.as_datatype_enum]:
                        int8_count += 1
                    elif dtype_val in [tf.float32.as_datatype_enum, tf.float16.as_datatype_enum]:
                        float32_count += 1
                    else:
                        other_count += 1
                # Fallback check for structural data paths
                elif "dtype" in node.attr:
                    total_tensors += 1
                    dtype_val = node.attr["dtype"].type
                    if dtype_val in [tf.int8.as_datatype_enum, tf.qint8.as_datatype_enum]:
                        int8_count += 1
                    elif dtype_val in [tf.float32.as_datatype_enum, tf.float16.as_datatype_enum]:
                        float32_count += 1
                    else:
                        other_count += 1
    else:
        print("Evaluating flat top-level graph nodes...")
        for node in meta_graph.graph_def.node:
            unique_ops.add(node.op)
            if "dtype" in node.attr:
                total_tensors += 1
                dtype_val = node.attr["dtype"].type
                if dtype_val in [tf.int8.as_datatype_enum, tf.qint8.as_datatype_enum]:
                    int8_count += 1
                elif dtype_val in [tf.float32.as_datatype_enum, tf.float16.as_datatype_enum]:
                    float32_count += 1
                else:
                    other_count += 1

    if total_tensors == 0:
        total_tensors = 1

    int8_pct = (int8_count / total_tensors) * 100
    float32_pct = (float32_count / total_tensors) * 100

    file_size_bytes = os.path.getsize(pb_file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    file_size_kb = file_size_bytes / 1024

    # 3. Format and Print the Required Output Layout
    print("\n========================================")
    print("        UNIQUE OPERATIONS LIST")
    print("========================================")
    for i, op in enumerate(sorted(unique_ops), 1):
        print(f"  {i}. {op}")
    print("----------------------------------------\n")

    print("========================================")
    print("        QUANTIZATION METRICS")
    print("========================================")
    print(f"Total Tensors (Tracked): {total_tensors}")
    print(f"Quantized INT8 Nodes:    {int8_count} ({int8_pct:.1f}%)")
    print(f"Floating-Point Float32:  {float32_count} ({float32_pct:.1f}%)")
    print(f"Other Nodes (e.g. i32):  {other_count}")
    print("----------------------------------------")
    print(f"SavedModel File Size:    {file_size_mb:.2f} MB ({file_size_kb:.2f} KB)")
    print("========================================")


def main(argv):
    del argv
    r = runfiles.Create()
    
    # 1. Resolve absolute sandbox paths from Bazel runfiles
    model_token = FLAGS.base_model.split()[0]  # 🟩 FIX: Extract index 0 to get a string
    abs_model_path = r.Rlocation(model_token)
    
    if not abs_model_path:
        raise FileNotFoundError(f"Bazel runfiles could not find token: {model_token}")
    
    # Extract the first image token to find where the validation folder lives
    all_image_tokens = FLAGS.calibration_images.split()
    if not all_image_tokens:
        raise ValueError("No calibration images found in the target build dependency list.")
    
    first_image_path = r.Rlocation(all_image_tokens[0])  # 🟩 FIX: Extract index 0 safely
    abs_calibration_dir = os.path.dirname(first_image_path)
    
    print(f"Loading Base Weights from: {abs_model_path}")
    print(f"Found Calibration Dataset Directory: {abs_calibration_dir}")
    
    # Enforce the .pt file extension for Ultralytics file parsing validators
    pt_model_path = os.path.join(os.getcwd(), "yolov8n.pt")
    if not os.path.exists(pt_model_path):
        shutil.copy(abs_model_path, pt_model_path)
    print(f"Aliased model path inside sandbox: {pt_model_path}")
    file_size_bytes = os.path.getsize(pt_model_path)
    print(f"Base model file size: {file_size_bytes / (1024 * 1024):.2f} MB ({file_size_bytes / 1024:.2f} KB)")

    # 2. Dynamically generate the data.yaml file required by YOLO calibration
    yaml_content = f"""
path: {os.path.dirname(abs_calibration_dir)}
train: {os.path.basename(abs_calibration_dir)}
val: {os.path.basename(abs_calibration_dir)}
names:
  0: person
"""
    yaml_path = "sandbox_calib.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    # 3. Initialize the model using the .pt path layout and execute strict full-integer quantization
    model = YOLO(pt_model_path)
    
    print("Exporting PyTorch weights to unquantized TensorFlow SavedModel...")
    # This automatically converts the model graph and creates a local directory
    saved_model_dir = model.export(format="saved_model")
    
    # 2. Extract the actual path string pointing to the generated saved_model.pb file
    pb_model_path = os.path.join(saved_model_dir, "saved_model.pb")
    print(f"Targeting generated ProtoBuf file: {pb_model_path}")
    
    # 3. Pass the valid string path to your file parser function
    analyze_pb_model(pb_model_path)
    
    print("Exporting to strict 100% pure INT8 TFLite format...")
    # exported_path_str = model.export(
    #     format="tflite",
    #     quantize=8,
    #     data=yaml_path,
    #     batch=1,
    #     fraction = 0.02
    # )
    
    # 4. Securely copy the generated file out of the sandbox back to your local folder
    workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace_dir:
        final_destination_tflite = pathlib.Path(workspace_dir) / "demo_applications/object_recognition/_build_dir/models/yolov8_int8.tflite"
        final_destination_pb = pathlib.Path(workspace_dir) / "demo_applications/object_recognition/_build_dir/models/yolov8_int8_saved_model.pb"
        final_destination_tflite = pathlib.Path(workspace_dir) / "demo_applications/object_recognition/_build_dir/models/yolov8_int8_quantized.tflite"
        final_destination_tflite.parent.mkdir(parents=True, exist_ok=True)
        final_destination_pb.parent.mkdir(parents=True, exist_ok=True)
        
        pb_source = os.path.join(saved_model_dir, "saved_model.pb")
        if not os.path.exists(pb_source):
            print(f"Error: Could not locate saved_model.pb at: {pb_source}")
        else:
            shutil.copy(pb_source, str(final_destination_pb))
            print(f"\n🟩 Success! SavedModel ProtoBuf copied to: {final_destination_pb}")
    #     if os.path.isdir(exported_path_str):
    #         tflite_source = os.path.join(exported_path_str, "yolov8n_int8.tflite")
    #         if not os.path.exists(tflite_source):
    #             found_files = glob.glob(os.path.join(exported_path_str, "*.tflite"))
    #             if found_files:
    #                 tflite_source = found_files[0]
    #         shutil.copy(tflite_source, str(final_destination))
    #     else:
    #         shutil.copy(exported_path_str, str(final_destination))
    #     print(f"\n🟩 Success! Pure INT8 model saved out of the sandbox to: {final_destination}")
    # else:
    #     print(f"Sandbox operation complete. Model built at temporary file path: {exported_path_str}")

if __name__ == '__main__':
    app.run(main)