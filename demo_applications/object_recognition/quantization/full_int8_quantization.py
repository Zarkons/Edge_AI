import os
import shutil
import pathlib
import cv2
import numpy as np
import argparse  # Native package to bypass absl dependency mismatches
from ultralytics import YOLO
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
from python.runfiles import runfiles

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

def main():
    # 2. Native Python Argument Parsing Configuration
    parser = argparse.ArgumentParser(description="YOLOv8 Quantization & Delta mAP Evaluation Pipeline")
    parser.add_argument('--base_model', required=True, help='Runfiles lookup key for the base .pt file.')
    parser.add_argument('--calibration_images', required=True, help='Runfiles space-separated list of image files.')
    parser.add_argument('--build_dir', default='demo_applications/object_recognition/quantization/_build_dir/models', 
                        help='Relative workspace path destination where all final models will be saved.')
    parser.add_argument('--validation_dataset_yaml', default='coco8.yaml', 
                        help='Dataset YAML configuration name for tracking mAP metrics (e.g., coco8.yaml).')
    
    args = parser.parse_args()
    r = runfiles.Create()
    
    # 3. Resolve absolute sandbox paths from Bazel runfiles
    model_token = args.base_model.split()[0]
    abs_model_path = r.Rlocation(model_token)
    
    if not abs_model_path:
        raise FileNotFoundError(f"Bazel runfiles could not find token: {model_token}")
    
    all_image_tokens = args.calibration_images.split()
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
    
    # 4. Stage 1: Load Base PyTorch Model and Run Validation Baseline
    model = YOLO(pt_model_path)
    
    print("\n--- CRITICAL METRIC PROFILE: Evaluating Baseline FP32 Performance ---")
    # Note: 'plots=False' avoids sandbox write errors during isolated Bazel runs
    validation_results = model.val(data=args.validation_dataset_yaml, plots=False, verbose=False)
    
    baseline_map50 = validation_results.box.map50
    baseline_map50_95 = validation_results.box.map  # This tracks standard mAP@0.5:0.95
    
    print(f"📊 BASELINE FP32 mAP@50:       {baseline_map50:.4f}")
    print(f"📊 BASELINE FP32 mAP@50:95:    {baseline_map50_95:.4f}")
    print("-------------------------------------------------------------------\n")
    
    # 5. Stage 2: Export PyTorch weights directly into standard FP32 ONNX
    print("Exporting PyTorch weights to unquantized FP32 ONNX...")
    fp32_onnx_path = model.export(format="onnx", imgsz=640, dynamic=False)
    print(f"Generated target FP32 ONNX file: {fp32_onnx_path}")
    
    # 6. Stage 3: Create calibration reader and execute static integer quantization
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
    
    # 7. Stage 4: Evaluate Quantized INT8 ONNX Accuracy to calculate performance degradation
    print("\n--- CRITICAL METRIC PROFILE: Evaluating Quantized INT8 ONNX Performance ---")
    try:
        # Load the newly compiled INT8 ONNX model straight into the validation loop
        int8_model = YOLO(int8_onnx_path, task="detect")
        int8_results = int8_model.val(data=args.validation_dataset_yaml, plots=False, verbose=False)
        
        int8_map50 = int8_results.box.map50
        int8_map50_95 = int8_results.box.map
        
        # Calculate the direct degradation delta for your thesis documentation
        delta_map50 = int8_map50 - baseline_map50
        delta_map50_95 = int8_map50_95 - baseline_map50_95
        
        print(f"📉 QUANTIZED INT8 mAP@50:    {int8_map50:.4f}  (Accuracy Delta: {delta_map50:+.4f})")
        print(f"📉 QUANTIZED INT8 mAP@50:95: {int8_map50_95:.4f}  (Accuracy Delta: {delta_map50_95:+.4f})")
    except Exception as e:
        print(f"⚠️ Could not complete direct INT8 ONNX validation: {e}")
    print("-------------------------------------------------------------------\n")
    
    # 8. Resolve destination path based on Bazel environment configurations
    workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    if workspace_dir:
        # User executed via 'bazel run' -> Write out to persistent workspace directory
        destination_dir = pathlib.Path(workspace_dir) / args.build_dir
    else:
        # Fallback if executing as part of an isolated build rule target
        destination_dir = pathlib.Path(os.getcwd()) / "exported_models"
        
    destination_dir.mkdir(parents=True, exist_ok=True)
    
    final_destination_pt = destination_dir / "yolov8n.pt"
    final_destination_fp32 = destination_dir / "yolov8n_fp32.onnx"
    final_destination_int8 = destination_dir / "yolov8n_int8.onnx"
    
    print(f">>> Streaming model outputs to targeting build directory: {destination_dir}")
    
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
    main()
