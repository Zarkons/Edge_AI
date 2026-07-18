#include "camera_input_handler.h"
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
#include "TracyProfiler.h"
#include "video_pipeline.h"
#include "ml_model_resolver.h"
#include "application_constants.h"
#include "print_logger.h"

using namespace obj_rec::app;
using namespace dsp::image;
using namespace ml::engine;

int main(int argc, char *argv[])
{
    /* Allocation Start */
    CameraInputHandler camera_handler;
    CameraFrame live_frame;
    LetterboxPreprocessor preprocessor;
    PlanarScaleTensorPacker packer;
    YOLOv8PostProcessor post_processor;
    ONNXRuntimeEngine engine;
    std::string abs_model_path;
    int64_t input_shape[] = {1, target_channels, target_height, target_width};
    size_t shape_dims = 4;
    size_t processed_counter = 0;

    VideoFrame frame_buffer;
    std::vector<uint8_t> preprocessed_buffer(target_width * target_height * target_channels);
    std::vector<float> tensor_buffer(target_width * target_height * target_channels);
    std::vector<float> out_preallocated_buffer(output_buffer_capacity);

    VideoPipeline<LetterboxPreprocessor, PlanarScaleTensorPacker, ONNXRuntimeEngine> pipeline(
        std::move(preprocessor),
        std::move(packer),
        std::move(engine),
        target_width,
        target_height,
        target_channels,
        preprocessed_buffer.data(),
        preprocessed_buffer.size(),
        tensor_buffer.data(),
        tensor_buffer.size(),
        input_shape,
        shape_dims,
        out_preallocated_buffer.data(),
        out_preallocated_buffer.size());
    /* Allocation End */
    PROFILER_LOG("Profiling enabled: Tracy Profiler active. Use Tracy GUI to visualize performance metrics.");

    try
    {
        PRINT_LOG("============= Live Object Recognition Application =============");

        abs_model_path = ResolveAbsModelPath(argv[0]);
        PRINT_LOG("[ObjRec][INFO] Resolved Model Path : " << abs_model_path);

        if (!pipeline.InitializeEngine(abs_model_path, intra_op_threads, inter_op_threads))
        {
            std::cerr << "[ObjRec][ERROR] Failed to initialize ONNX Runtime engine." << std::endl;
            return -1;
        }
        PRINT_LOG("[ObjRec][INFO] ONNX Engine initialized successfully.");

        std::vector<std::string> model_classes = pipeline.GetClassNames();
        if (model_classes.empty())
        {
            PRINT_LOG("[ObjRec][ERROR] Could not resolve embedded model metadata strings. Falling back to numeric Class IDs.");
        }
        else
        {
            PRINT_LOG("[ObjRec][INFO] Successfully parsed and initialized [" << model_classes.size() << "] distinct text class tags directly from the ONNX binary.");
        }

        if (!camera_handler.Initialize(std::string{kIphoneStreamUrl}))
        {
            std::cerr << "[ObjRec][ERROR] Could not connect to live iPhone stream pipeline." << std::endl;
            return -1;
        }

        PRINT_LOG("🚀 Starting Live iPhone Inference Loop. Press 'q' inside the window canvas to exit.");

        /* Hot-Path Inference Loop, no dynamic memory allocations after this point! */
        bool next_frame_available = true;
        while (next_frame_available)
        {
            ZoneScopedN("Live Frame Loop");
            next_frame_available = camera_handler.GetNextFrame(live_frame);
            processed_counter++;

            // Map incoming live frame buffers directly into structural signal processing dimensions
            frame_buffer.data_ptr = live_frame.data.data();
            frame_buffer.width = live_frame.width;
            frame_buffer.height = live_frame.height;
            frame_buffer.channels = live_frame.channels;
            frame_buffer.stride = live_frame.stride;

            PRINT_LOG("-> Processing Live Camera Frame [" << processed_counter << "]");
            {
                ZoneScopedN("Preprocessing");
                if (!pipeline.StreamFramePreprocess(frame_buffer))
                    continue;
            }
            {
                ZoneScopedN("Inference");
                if (!pipeline.InferenceExecute())
                {
                    std::cerr << "   [Inference Failure] Engine run failed for frame [" << processed_counter << "]" << std::endl;
                    if (cv::waitKey(1) == 'q')
                        break;
                    continue;
                }
            }

            /* Postprocessing to be moved to a separate thread/application */
            TransformMetadata meta = pipeline.GetLatestTransformMetadata();
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

            PRINT_LOG("   [Detections Filtered]: " << detections.size() << " objects tracked.");
            for (size_t i = 0; i < detections.size(); ++i)
            {
                std::string class_name = (detections[i].class_id >= 0 && static_cast<size_t>(detections[i].class_id) < model_classes.size())
                                             ? model_classes[detections[i].class_id]
                                             : "Unknown";

                PRINT_LOG("      Object [" << i + 1 << "] -> " << class_name
                                           << " | Conf: " << (detections[i].confidence * 100.0f) << "%"
                                           << " | Mapped Box: ["
                                           << detections[i].x1 << ", " << detections[i].y1 << ", "
                                           << detections[i].x2 << ", " << detections[i].y2 << "]");
            }

            // FIX: Pass 1ms instead of 0 to allow the live video stream to decode and paint continuously.
            // Breaks execution cleanly when hitting 'q' inside the active visual window canvas.
            if (cv::waitKey(1) == 'q')
            {
                PRINT_LOG("[INFO] 'q' key pressed. Stopping live stream loop cleanly.");
                break;
            }

            else
            {
            }
            PRINT_LOG("---------------------------------------------------------");
        }

        PRINT_LOG("🟩 Live Execution Terminated cleanly. Total frames processed: " << processed_counter);
    }

    catch (const Ort::Exception &e)
    {
        std::cerr << "[ONNX][Run] ORT exception: " << e.what() << std::endl;
        return false;
    }
    catch (const std::exception &e)
    {
        std::cerr << "🟥 Critical Exception Aborted Operation: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
