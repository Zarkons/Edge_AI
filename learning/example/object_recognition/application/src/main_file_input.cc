#include "file_input_handler.h"
#include "dip_data_types.h"
#include "letterbox_preprocessor.h"
#include <vector>
#include "planar_scale_tensor_packer.h"
#include "onnxruntime_engine.h"
#include "yolov8_post_processor.h"
#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "yolov8_visualizer.h"

constexpr int32_t target_width = 640;
constexpr int32_t target_height = 640;
constexpr int32_t target_channels = 3;
constexpr size_t ouput_buffer_capacity = 800000; // Pre-allocate a large enough buffer for the output tensor

using namespace obj_rec::app;
using namespace dsp::image;
using namespace inference_engines::onnxruntime;

int main(int argc, char *argv[])
{
    // Hardcode your runfiles lookup tokens directly.
    // Bazel maps '@repo_name//file' to 'repo_name/file/<filename>' inside runfiles.
    // If the repository name has an embedded plus due to Bzlmod, use the canonical name.
    std::string model_token = "learning/example/object_recognition/_build_dir/models/yolov8n_fp32.onnx";
    std::string calibration_token = "coco_val2017/val2017/000000000139.jpg";

    // Optional override: set OBJREC_MODEL_VARIANT=int8 to use quantized model.
    const char *model_variant = std::getenv("OBJREC_MODEL_VARIANT");
    if (model_variant != nullptr && std::string(model_variant) == "int8")
    {
        model_token = "learning/example/object_recognition/_build_dir/models/yolov8n_int8.onnx";
    }

    try
    {
        std::cout << "============= Object Recognition Application =============" << std::endl;

        // Instantiate FileInputHandler passing argv[0] purely for bootstrapping runfiles directories
        obj_rec::app::FileInputHandler input_handler(argv[0], calibration_token);

        // Resolve the model path completely via the hardcoded data string
        std::string abs_model_path = input_handler.GetModelPath(model_token);

        std::cout << "[INFO] FileInputHandler initialized successfully." << std::endl;
        std::cout << "[INFO] Resolved Model Path via data: " << abs_model_path << std::endl;
        std::cout << "[INFO] Total discoverable calibration images: "
                  << input_handler.GetTotalImagesCount() << "\n"
                  << std::endl;

        // Rest of your working while(input_handler.GetNext(disk_frame)) loops remain exactly the same...
        dsp::image::VideoFrame frame_buffer;
        obj_rec::app::RawBufferFrame disk_frame;
        LetterboxPreprocessor preprocessor;
        PlanarScaleTensorPacker packer(true);
        YOLOv8PostProcessor post_processor;

        std::vector<uint8_t> preprocessed_buffer(target_width * target_height * target_channels);
        std::vector<float> tensor_buffer(target_width * target_height * target_channels);

        size_t max_output_elements = ouput_buffer_capacity;
        std::vector<float> out_preallocated_buffer(max_output_elements);
        std::vector<int64_t> output_shape_memory(4);

        InferenceOutput output_result;
        output_result.shape = output_shape_memory.data();
        output_result.shape_capacity = output_shape_memory.size();

        // Initialize the ONNX Runtime engine ONCE outside the loop
        ONNXRuntimeEngine engine;
        if (!engine.Initialize(abs_model_path, 1, 1))
        {
            std::cerr << "Failed to initialize ONNX Runtime engine." << std::endl;
            return -1;
        }
        std::cout << "[INFO] ONNX Engine initialized successfully.\n"
                  << std::endl;

        std::vector<std::string> model_classes = engine.GetClassNames();

        if (model_classes.empty())
        {
            std::cout << "[WARN] Could not resolve embedded model metadata strings. Falling back to numeric Class IDs." << std::endl;
        }
        else
        {
            std::cout << "[INFO] Successfully parsed and initialized [" << model_classes.size()
                      << "] distinct text class tags directly from the ONNX binary." << std::endl;
        }

        size_t processed_counter = 0;
        while (input_handler.GetNext(disk_frame) && processed_counter < 10)
        {
            processed_counter++;

            frame_buffer.data_ptr = disk_frame.data.data();
            frame_buffer.width = disk_frame.width;
            frame_buffer.height = disk_frame.height;
            frame_buffer.channels = disk_frame.channels;
            frame_buffer.stride = disk_frame.stride;

            std::cout << "-> Processing Frame [" << processed_counter << "]" << std::endl;

            TransformMetadata meta = preprocessor.Execute(frame_buffer, target_width, target_height, preprocessed_buffer.data(), preprocessed_buffer.size());
            packer.Pack(preprocessed_buffer.data(), target_width, target_height, target_channels, tensor_buffer.data(), tensor_buffer.size());

            int64_t input_shape[] = {1, target_channels, target_height, target_width};
            size_t shape_dims = 4;

            bool success = engine.Run(
                tensor_buffer.data(),
                input_shape,
                shape_dims,
                out_preallocated_buffer.data(),
                out_preallocated_buffer.size(),
                output_result);

            if (success)
            {
                std::cout << "   [Inference Success] Elements: " << output_result.element_count << " | Output Shape: [";
                for (size_t i = 0; i < output_result.shape_dims; ++i)
                {
                    std::cout << output_result.shape[i] << (i < output_result.shape_dims - 1 ? "x" : "");
                }
                std::cout << "]" << std::endl;

                std::vector<Detection> detections = post_processor.Execute(
                    out_preallocated_buffer.data(),
                    0.25f, // Confidence Threshold
                    0.45f, // NMS Overlap IoU Threshold
                    meta   // Transfrom metrics containing source sizing bounds
                );

                cv::Mat frame(disk_frame.height, disk_frame.width, CV_8UC3, disk_frame.data.data(), disk_frame.stride);

                cv::Mat visualization_frame = frame.clone();

                obj_rec::app::YOLOv8Visualizer::DrawDetections(frame, detections, model_classes);

                // 3. Display the visualization window onto your desktop environment
                cv::imshow("YOLOv8 FP32 Detections Canvas", frame);

                // 4. Crucial: wait 1ms (for videos) or 0 (indefinitely for static images) to allow UI thread to paint
                cv::waitKey(0);

                // Optional: Save the visualized frame to disk
                cv::imwrite("output_visualized.jpg", frame);

                std::cout << "   [Detections Filtered]: " << detections.size() << " objects tracked." << std::endl;
                for (size_t i = 0; i < detections.size(); ++i)
                {
                    std::cout << "      Object [" << i + 1 << "] -> Class ID: " << detections[i].class_id
                              << " | Conf: " << (detections[i].confidence * 100.0f) << "%"
                              << " | Mapped Box (Original Pixels): ["
                              << detections[i].x1 << ", " << detections[i].y1 << ", "
                              << detections[i].x2 << ", " << detections[i].y2 << "]" << std::endl;
                }
            }
            else
            {
                std::cerr << "   [Inference Failure] Engine run failed for frame [" << processed_counter << "]" << std::endl;
            }
            std::cout << "---------------------------------------------------------" << std::endl;
        }

        std::cout << "\n🟩 Execution Complete. Successfully processed " << processed_counter << " files." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "🟥 Critical Exception Aborted Operation: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
