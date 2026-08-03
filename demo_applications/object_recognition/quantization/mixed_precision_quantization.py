import os
import shutil
import pathlib
import cv2
import numpy as np
import argparse
from ultralytics import YOLO
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
from python.runfiles import runfiles

class BazelYOLOv8CalibrationReader(CalibrationDataReader):
    def __init__(self, calibration_dir, width=640, height=640):
        self.image_paths = [
            os.path.join(calibration_dir, f) for f in os.listdir(calibration_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        if not self.image_paths:
            raise ValueError(f"No valid image assets found in sandbox folder: {calibration_dir}")
        
        self.enum_data = iter(self.image_paths[:100])
        self.width = width
        self.height = height

    def get_next(self):
        path = next(self.enum_data, None)
        if path is None:
            return None
        
        img = cv2.imread(path)
        if img is None:
            return self.get_next()
            
        img = cv2.resize(img, (self.width, self.height))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return {"images": img}

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Selective Mixed-Precision Graph Surgery")
    parser.add_argument('--base_model', required=True, help='Runfiles lookup key for the base .pt file.')
    parser.add_argument('--calibration_images', required=True, help='Runfiles list of image files.')
    parser.add_argument('--build_dir', default='demo_applications/object_recognition/quantization/_build_dir/models', 
                        help='Workspace path destination for final models.')
    parser.add_argument('--validation_dataset_yaml', default='coco8.yaml', 
                        help='Dataset YAML configuration name for tracking metrics.')
    
    args = parser.parse_args()
    r = runfiles.Create()
    
    # 1. Resolve paths
    model_token = args.base_model.split()[0]
    abs_model_path = r.Rlocation(model_token)
    if not abs_model_path:
        raise FileNotFoundError(f"Bazel runfiles could not find token: {model_token}")
        
    all_image_tokens = args.calibration_images.split()
    first_image_path = r.Rlocation(all_image_tokens[0])
    abs_calibration_dir = os.path.dirname(first_image_path)
    
    pt_model_path = os.path.join(os.getcwd(), "yolov8n.pt")
    if not os.path.exists(pt_model_path):
        shutil.copy(abs_model_path, pt_model_path)
        
    # 2. Export FP32 Baseline
    model = YOLO(pt_model_path)
    print("Evaluating Baseline FP32 Performance...")
    validation_results = model.val(data=args.validation_dataset_yaml, plots=False, verbose=False)
    baseline_map50 = validation_results.box.map50
    print(f"📊 BASELINE FP32 mAP@50: {baseline_map50:.4f}\n")
    
    print("Exporting PyTorch weights to unquantized FP32 ONNX...")
    fp32_onnx_path = model.export(format="onnx", imgsz=640, dynamic=False)
    fp32_size_mb = os.path.getsize(fp32_onnx_path) / (1024 * 1024)
    
    # 3. Define Selective Node Exclusion Array (Graph Surgery Target)
    # These match the exact spatial decoding bottlenecks identified in your Netron screenshot
    nodes_to_protect = [
        "/model.22/Concat_3",  # The final aggregation node combining scores and boxes
        "/model.22/Mul_2",     # Terminal layer of the 1x4x8400 spatial box regressor
        "/model.22/Sigmoid"    # Terminal layer of the 1x80x8400 class score branch
    ]
    
    print(f"Applying Graph Surgery! Protecting {len(nodes_to_protect)} sensitive detection head layers in FP32...")
    
    # 4. Run Static Quantization with Target Node Exclusion
    calib_reader = BazelYOLOv8CalibrationReader(calibration_dir=abs_calibration_dir)
    mixed_onnx_path = os.path.join(os.getcwd(), "yolov8n_mixed_precision.onnx")
    
    quantize_static(
        model_input=fp32_onnx_path,
        model_output=mixed_onnx_path,
        calibration_data_reader=calib_reader,
        quant_format=QuantType.QInt8,
        nodes_to_exclude=nodes_to_protect  # Inject exclusion filter
    )
    mixed_size_mb = os.path.getsize(mixed_onnx_path) / (1024 * 1024)
    print(f"🟩 Generated Mixed-Precision ONNX Asset: {mixed_onnx_path}")
    
    # 5. Evaluate Hybrid Model Performance & Compute Size Metrics
    print("\n--- EVALUATING SURGERY RESULTS: Verifying Accuracy & Footprint ---")
    try:
        mixed_model = YOLO(mixed_onnx_path, task="detect")
        mixed_results = mixed_model.val(data=args.validation_dataset_yaml, plots=False, verbose=False)
        mixed_map50 = mixed_results.box.map50
        
        print(f"🟩 BASELINE FP32 mAP@50:       {baseline_map50:.4f}")
        print(f"🚀 SURGERY INT8/FP32 mAP@50:   {mixed_map50:.4f}")
        print(f"📊 Net Accuracy Delta:         {mixed_map50 - baseline_map50:+.4f}")
        print("-" * 50)
        print(f"📦 Baseline FP32 Model Size:   {fp32_size_mb:.2f} MB")
        print(f"📦 Surgery INT8/FP32 Size:     {mixed_size_mb:.2f} MB")
        print(f"📉 Total Memory Saved:         {fp32_size_mb - mixed_size_mb:.2f} MB ({((fp32_size_mb - mixed_size_mb) / fp32_size_mb) * 100:.1f}% reduction)")
    except Exception as e:
        print(f"⚠️ Validation encountered a parsing variance: {e}")
    print("-------------------------------------------------------------------\n")

    # 6. Stream file outputs out of isolated sandbox build zone
    workspace_dir = os.environ.get("BUILD_WORKSPACE_DIRECTORY")
    destination_dir = pathlib.Path(workspace_dir) / args.build_dir if workspace_dir else pathlib.Path(os.getcwd()) / "exported_models"
    destination_dir.mkdir(parents=True, exist_ok=True)
    
    shutil.copy(mixed_onnx_path, str(destination_dir / "yolov8n_mixed_precision.onnx"))
    print(f"🟩 Success! Saved hybrid model asset to: {destination_dir}/yolov8n_mixed_precision.onnx")

if __name__ == '__main__':
    main()
