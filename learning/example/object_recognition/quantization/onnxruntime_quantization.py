import os
import shutil
import pathlib
import cv2
import numpy as np
from ultralytics import YOLO
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
from python.runfiles import runfiles
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string('base_model', None, 'Runfiles lookup key for the base .pt file.')
flags.DEFINE_string('calibration_images', None, 'Runfiles space-separated list of image files.')
flags.DEFINE_string('build_dir', 'learning/example/object_recognition/_build_dir/models', 
                    'Relative workspace path destination where all final models will be saved.')

# 1. Custom Calibration Reader to stream data from your Bazel runfiles directory
class BazelYOLOv8CalibrationReader(CalibrationDataReader):
    def __init__(self, calibration_dir, width=640, height=640):
        # Gather all images found inside the sandbox calibration folder
        self.image_paths = [
            os.path.join(calibration_dir, f) for f in os.listdir(calibration_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if not self.image_paths:
            raise ValueError(f"No valid image assets (.jpg/.png) found in sandbox folder: {calibration_dir}")
        
        # Limit to 50-100 images to keep the Bazel execution sandbox run swift
        self.enum_data = iter(self.image_paths[:100])
        self.width = width
        self.height = height

    def get_next(self):
        path = next(self.enum_data, None)
        if path is None:
            return None
        
        # Preprocess frame explicitly matching YOLOv8 configuration rules
        img = cv2.imread(path)
        if img is None:
            return self.get_next() # Skip corrupt files safely
            
        img = cv2.resize(img, (self.width, self.height))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1)) # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Shape: (1, 3, 640, 640)
        
        # 'images' is the explicit input tensor node identifier for YOLOv8 graphs
        return {"images": img}

def main(argv):
    del argv
    r = runfiles.Create()
    
    # 1. Resolve absolute sandbox paths from Bazel runfiles
    model_token = FLAGS.base_model.split()[0]
    abs_model_path = r.Rlocation(model_token)
    
    if not abs_model_path:
        raise FileNotFoundError(f"Bazel runfiles could not find token: {model_token}")
    
    # Extract the first image token to find where the validation folder lives
    all_image_tokens = FLAGS.calibration_images.split()
    if not all_image_tokens:
        raise ValueError("No calibration images found in the target build dependency list.")
    
    first_image_path = r.Rlocation(all_image_tokens[0])
    abs_calibration_dir = os.path.dirname(first_image_path)
    
    print(f"Loading Base Weights from: {abs_model_path}")
    print(f"Found Calibration Dataset Directory: {abs_calibration_dir}")
    
    # Enforce the .pt file extension for Ultralytics file parsing validators
    pt_model_path = os.path.join(os.getcwd(), "yolov8n.pt")
    if not os.path.exists(pt_model_path):
        shutil.copy(abs_model_path, pt_model_path)
    print(f"Aliased model path inside sandbox: {pt_model_path}")
    
    # 2. Stage 1: Export PyTorch directly into standard FP32 ONNX
    model = YOLO(pt_model_path)
    print("Exporting PyTorch weights to unquantized FP32 ONNX...")
    
    # Generates a standard 'yolov8n.onnx' file right inside the local sandbox run directory
    fp32_onnx_path = model.export(format="onnx", imgsz=640, dynamic=False)
    print(f"Generated target FP32 ONNX file: {fp32_onnx_path}")
    
    # 3. Stage 2: Create calibration reader and execute static integer quantization
    print("Initializing Calibration Data Stream Reader...")
    calib_reader = BazelYOLOv8CalibrationReader(calibration_dir=abs_calibration_dir)
    
    int8_onnx_path = os.path.join(os.getcwd(), "yolov8n_int8.onnx")
    print(f"Executing native ONNX Runtime Static Quantization -> {int8_onnx_path}")
    
    quantize_static(
        model_input=fp32_onnx_path,
        model_output=int8_onnx_path,
        calibration_data_reader=calib_reader,
        quant_format=QuantType.QInt8 # Signed 8-bit integer processing pipeline
    )
    
    # 4. Resolve destination path based on Bazel environment
    workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace_dir:
        # User executed via 'bazel run' -> Write out to persistent workspace directory
        destination_dir = pathlib.Path(workspace_dir) / FLAGS.build_dir
    else:
        # Fallback if executing as part of an isolated build rule target
        destination_dir = pathlib.Path(os.getcwd()) / "exported_models"
        
    destination_dir.mkdir(parents=True, exist_ok=True)
    
    final_destination_pt = destination_dir / "yolov8n.pt"
    final_destination_fp32 = destination_dir / "yolov8n_fp32.onnx"
    final_destination_int8 = destination_dir / "yolov8n_int8.onnx"
    
    print(f"\n>>> Streaming model outputs to targeting build directory: {destination_dir}")
    
    # Save the original PyTorch model asset
    shutil.copy(pt_model_path, str(final_destination_pt))
    print(f"🟩 Success! Base PyTorch Weights saved to: {final_destination_pt}")
    
    # Copy out the baseline unquantized ONNX file
    if os.path.exists(fp32_onnx_path):
        shutil.copy(fp32_onnx_path, str(final_destination_fp32))
        print(f"🟩 Success! Unquantized FP32 ONNX saved to: {final_destination_fp32}")
        
    # Copy out the compiled INT8 ONNX optimization file
    if os.path.exists(int8_onnx_path):
        shutil.copy(int8_onnx_path, str(final_destination_int8))
        print(f"🟩 Success! Optimized INT8 QNX-ready ONNX saved to: {final_destination_int8}")
        
        # Print file summary metrics
        int8_size = os.path.getsize(int8_onnx_path)
        print(f"Final Quantized Asset File Size: {int8_size / (1024 * 1024):.2f} MB")

if __name__ == '__main__':
    app.run(main)
