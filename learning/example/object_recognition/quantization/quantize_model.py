import os
import pathlib
import sys
import shutil # 🟩 Added for file copying
import glob
from absl import app, flags
from ultralytics import YOLO
from python.runfiles import runfiles

FLAGS = flags.FLAGS
flags.DEFINE_string('base_model', None, 'Runfiles lookup key for the base .pt file.')
flags.DEFINE_string('calibration_images', None, 'Runfiles space-separated list of image files.')

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

    # 2. Dynamically generate the data.yaml file required by YOLO calibration
    yaml_content = f"""
path: {os.path.dirname(abs_calibration_dir)}
train: val2017
val: val2017
names:
  0: person
"""
    yaml_path = "sandbox_calib.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    # 3. Initialize the model using the .pt path layout and execute strict full-integer quantization
    model = YOLO(pt_model_path)
    
    print("Exporting to strict 100% pure INT8 TFLite format...")
    exported_path_str = model.export(
        format="tflite",
        quantize=8,
        data=yaml_path,
        batch=1,
        fraction = 0.02
    )
    
    # 4. Securely copy the generated file out of the sandbox back to your local folder
    workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace_dir:
        final_destination = pathlib.Path(workspace_dir) / "learning/example/object_recognition/quantized_model/yolov8_int8.tflite"
        final_destination.parent.mkdir(parents=True, exist_ok=True)
        
        if os.path.isdir(exported_path_str):
            tflite_source = os.path.join(exported_path_str, "yolov8n_int8.tflite")
            if not os.path.exists(tflite_source):
                found_files = glob.glob(os.path.join(exported_path_str, "*.tflite"))
                if found_files:
                    tflite_source = found_files[0]
            shutil.copy(tflite_source, str(final_destination))
        else:
            shutil.copy(exported_path_str, str(final_destination))
        print(f"\n🟩 Success! Pure INT8 model saved out of the sandbox to: {final_destination}")
    else:
        print(f"Sandbox operation complete. Model built at temporary file path: {exported_path_str}")

if __name__ == '__main__':
    app.run(main)