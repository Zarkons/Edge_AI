#include "CameraInputHandler.h"
#include "FileInputHandler.h"
#include "dip_data_types.h"
#include "LetterboxPreprocessor.h"
#include <vector>
#include "PlanarScaleTensorPacker.h"
#include "ONNXRuntimeEngine.h"
#include "YOLOv8PostProcessor.h"
#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "YOLOv8Visualizer.h"
#include "TracyProfiler.h"

#include "tools/cpp/runfiles/runfiles.h"
using bazel::tools::cpp::runfiles::Runfiles;

constexpr int32_t target_width = 640;
constexpr int32_t target_height = 640;
constexpr int32_t target_channels = 3;
constexpr size_t ouput_buffer_capacity = 800000; // Pre-allocate a large enough buffer for the output tensor

using namespace obj_rec::app;
using namespace dsp::image;
using namespace onnxruntime_engine::inference;

int main(int argc, char *argv[])
{
    PROFILER_LOG("Profiling enabled: Tracy Profiler active. Use Tracy GUI to visualize performance metrics.");
    // Hardcode your runfiles lookup tokens directly.
    std::string model_token = "_main/learning/example/object_recognition/quantization/_build_dir/models/yolov8n_fp32.onnx";

    try
    {
        std::cout << "============= Live Object Recognition Application =============" << std::endl;

        // Optional override: set OBJREC_MODEL_VARIANT=int8 to use quantized model.
        std::string runfiles_error;
        std::unique_ptr<Runfiles> runfiles(Runfiles::Create(argv[0], &runfiles_error));
        if (!runfiles)
        {
            std::cerr << "Failed to init Bazel Runfiles: " << runfiles_error << std::endl;
            return -1;
        }

        // 2. Direct Lookup: Rely completely on your model token!
        std::string abs_model_path = runfiles->Rlocation(model_token);
        std::cout << "[INFO] Resolved Model Path directly: " << abs_model_path << std::endl;

        // Initialize modules
        CameraInputHandler camera_handler;
        dsp::image::VideoFrame frame_buffer;
        CameraFrame live_frame;
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

        // Extract class metadata strings directly from your exported ONNX file header
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

        // Configure connection string for your iPhone's IP Cam App stream
        // Update this string with the matching IP/Port displayed inside your phone's app interface!
        std::string iphone_stream_url = "http://192.168.1.154:8080/stream.mjpeg";
        if (!camera_handler.Initialize(iphone_stream_url))
        {
            std::cerr << "CRITICAL: Could not connect to live iPhone stream pipeline." << std::endl;
            return -1;
        }

        size_t processed_counter = 0;
        std::cout << "\n🚀 Starting Live iPhone Inference Loop. Press 'q' inside the window canvas to exit.\n"
                  << std::endl;

        // Continuous streaming loop fed directly from your camera hardware
        while (camera_handler.GetNextFrame(live_frame))
        {
            processed_counter++;

            // Map incoming live frame buffers directly into structural signal processing dimensions
            frame_buffer.data_ptr = live_frame.data.data();
            frame_buffer.width = live_frame.width;
            frame_buffer.height = live_frame.height;
            frame_buffer.channels = live_frame.channels;
            frame_buffer.stride = live_frame.stride;

            std::cout << "-> Processing Live Camera Frame [" << processed_counter << "]" << std::endl;
            TransformMetadata meta;

            {
                ZoneScopedN("Preprocessing");
                meta = preprocessor.Execute(frame_buffer, target_width, target_height, preprocessed_buffer.data(), preprocessed_buffer.size());
            }

            {
                ZoneScopedN("Packing");
                packer.Pack(preprocessed_buffer.data(), target_width, target_height, target_channels, tensor_buffer.data(), tensor_buffer.size());
            }

            int64_t input_shape[] = {1, target_channels, target_height, target_width};
            size_t shape_dims = 4;
            bool success = false;

            {
                ZoneScopedN("ONNX Inference");
                success = engine.Run(
                    tensor_buffer.data(),
                    input_shape,
                    shape_dims,
                    out_preallocated_buffer.data(),
                    out_preallocated_buffer.size(),
                    output_result);
            }

            if (success)
            {
                std::vector<Detection> detections;
                {
                    ZoneScopedN("Postprocessing");
                    detections = post_processor.Execute(
                        out_preallocated_buffer.data(),
                        0.75f, // Confidence Threshold
                        0.45f, // NMS Overlap IoU Threshold
                        meta   // Transform metrics containing source sizing bounds
                    );
                }
                FrameMark;

                // Initialize OpenCV container directly referencing the incoming camera frame array
                cv::Mat frame(live_frame.height, live_frame.width, CV_8UC3, live_frame.data.data(), live_frame.stride);
                cv::Mat visualization_frame = frame.clone();

                // Draw localized text strings and bounding boxes onto your cloned matrix canvas
                obj_rec::app::YOLOv8Visualizer::DrawDetections(visualization_frame, detections, model_classes);

                // Display your updated visualization frame
                cv::imshow("YOLOv8 Live iPhone Cam Canvas", visualization_frame);

                // Optional: Save latest processed artifact file directly onto disk
                cv::imwrite("output_visualized.jpg", visualization_frame);

                std::cout << "   [Detections Filtered]: " << detections.size() << " objects tracked." << std::endl;
                for (size_t i = 0; i < detections.size(); ++i)
                {
                    std::string class_name = (detections[i].class_id >= 0 && static_cast<size_t>(detections[i].class_id) < model_classes.size())
                                                 ? model_classes[detections[i].class_id]
                                                 : "Unknown";

                    std::cout << "      Object [" << i + 1 << "] -> " << class_name
                              << " | Conf: " << (detections[i].confidence * 100.0f) << "%"
                              << " | Mapped Box: ["
                              << detections[i].x1 << ", " << detections[i].y1 << ", "
                              << detections[i].x2 << ", " << detections[i].y2 << "]" << std::endl;
                }

                // FIX: Pass 1ms instead of 0 to allow the live video stream to decode and paint continuously.
                // Breaks execution cleanly when hitting 'q' inside the active visual window canvas.
                if (cv::waitKey(1) == 'q')
                {
                    std::cout << "[INFO] 'q' key pressed. Stopping live stream loop cleanly." << std::endl;
                    break;
                }
            }
            else
            {
                std::cerr << "   [Inference Failure] Engine run failed for frame [" << processed_counter << "]" << std::endl;

                // Keep the UI rendering pipeline responsive even if a single network packet drops
                if (cv::waitKey(1) == 'q')
                    break;
            }
            std::cout << "---------------------------------------------------------" << std::endl;
        }

        std::cout << "\n🟩 Live Execution Terminated cleanly. Total frames processed: " << processed_counter << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "🟥 Critical Exception Aborted Operation: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
