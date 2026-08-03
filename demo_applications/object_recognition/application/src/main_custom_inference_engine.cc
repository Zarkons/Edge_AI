#include "camera_input_handler.h"
#include "file_input_handler.h"
#include "dip_data_types.h"
#include "letterbox_preprocessor.h"
#include <vector>
#include "planar_scale_tensor_packer.h"
#include "yolov8_post_processor.h"
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include "yolov8_visualizer.h"
#include "TracyProfiler.h"
#include "video_pipeline.h"
#include "ml_model_resolver.h"
#include "application_constants.h"
#include "print_logger.h"
#include "frame_handler.h"
#include "inference_pipeline.h"

#include <filesystem>
#include <system_error>

using namespace obj_rec::app;
using namespace dsp::image;
using namespace ml::engine;

int main(int argc, char *argv[])
{
    /* Allocation Start */
    // FrameHandler frame_handler;
    // CameraInputHandler camera_handler;
    // CameraFrame live_frame;
    // LetterboxPreprocessor preprocessor;
    // PlanarScaleTensorPacker packer;
    // YOLOv8PostProcessor post_processor;
    // std::string abs_model_path;
    // int64_t input_shape[] = {1, target_channels, target_height, target_width};
    // size_t shape_dims = 4;
    // size_t processed_counter = 0;
    InferencePipeline edge_engine;
    std::string weights_path = "demo_applications/object_recognition/quantization/_build_dir/compiled_engine_assets/yolov8n_weights.bin";
    std::string manifest_json_path = "demo_applications/object_recognition/quantization/_build_dir/compiled_engine_assets/yolov8n_manifest.json";

    // VideoFrame frame_buffer;
    // std::vector<uint8_t> preprocessed_buffer(target_width * target_height * target_channels);
    // std::vector<float> tensor_buffer(target_width * target_height * target_channels);
    // std::vector<float> out_preallocated_buffer(output_buffer_capacity);

    // VideoPipeline<LetterboxPreprocessor, PlanarScaleTensorPacker, InferencePipeline> pipeline(
    //     std::move(preprocessor),
    //     std::move(packer),
    //     std::move(edge_engine),
    //     target_width,
    //     target_height,
    //     target_channels,
    //     preprocessed_buffer.data(),
    //     preprocessed_buffer.size(),
    //     tensor_buffer.data(),
    //     tensor_buffer.size(),
    //     input_shape,
    //     shape_dims,
    //     out_preallocated_buffer.data(),
    //     out_preallocated_buffer.size());
    /* Allocation End */
    try
    {
        std::string resolved_weights_path = ResolveAbsWeightPath(argv[0]);
        if (!resolved_weights_path.empty())
        {
            weights_path = resolved_weights_path;
        }

        std::string resolved_manifest_path = ResolveAbsManifestPath(argv[0]);
        if (!resolved_manifest_path.empty())
        {
            manifest_json_path = resolved_manifest_path;
        }

        if (edge_engine.initialize_weight_buffer(weights_path))
        {
            PRINT_INFO("🟩 Custom Inference Engine initialized successfully with weight buffer size: "
                       << edge_engine.get_buffer_size() << " bytes.");
        }
        else
        {
            PRINT_ERROR("🟥 Failed to initialize Custom Inference Engine with weight buffer.");
            return -1;
        }

        std::vector<WeightTensorView> model_parameters_registry;
        std::vector<ExecutionStep> pruned_execution_graph;
        std::cout << "\n[Step 2/2] Parsing Topological Map & Applying Kernel Fusion...\n";
        if (!edge_engine.load_and_optimize_manifest(manifest_json_path,
                                                    model_parameters_registry,
                                                    pruned_execution_graph))
        {
            std::cerr << "🟥 CRITICAL COMPILER FAULT: Failed to decode or optimize JSON manifest.\n";
            return -1;
        }
        edge_engine.export_optimized_pipeline("optimized_execution_graph.json", pruned_execution_graph);

        const std::filesystem::path manifest_output_path = std::filesystem::path(manifest_json_path).parent_path() / "optimized_execution_graph.json";
        if (!edge_engine.export_optimized_pipeline(manifest_output_path.string(), pruned_execution_graph))
        {
            std::cerr << "🟥 Failed to export optimized execution graph beside manifest: "
                      << manifest_output_path << "\n";
            return -1;
        }

        // 4. Verification Check: Confirm your runtime state machine is fully loaded
        std::cout << "\n===================================================================\n";
        std::cout << "🟩 ENGINE INITIALIZATION VERIFICATION SUCCESSFUL\n";
        std::cout << "===================================================================\n";
        std::cout << "   ↳ Memory Map Status:     " << (edge_engine.ready() ? "ACTIVE" : "INACTIVE") << "\n"
                  << "   ↳ Total Mapped Weights:  " << (edge_engine.get_buffer_size() / (1024.0 * 1024.0)) << " MB\n"
                  << "   ↳ Virtual Base Address:  " << edge_engine.get_base_address() << "\n"
                  << "   ↳ Tracked Weight Views:  " << model_parameters_registry.size() << " tensors linked\n"
                  << "   ↳ Optimized Graph Size:  " << pruned_execution_graph.size() << " math steps retained\n";
        std::cout << "-------------------------------------------------------------------\n\n";

        // 5. Sample Pipeline Inspection: Trace the first operational layer configuration
        if (!pruned_execution_graph.empty())
        {
            const auto &first_step = pruned_execution_graph[0];
            const std::string first_input = first_step.inputs.empty() ? "<none>" : first_step.inputs[0];
            std::cout << "🔍 Inspecting Execution Graph Step 0 (The First Real Math Core Layer):\n"
                      << "   ↳ Node Type:           " << first_step.op_type << "\n"
                      << "   ↳ Node Identifier:     " << first_step.node_name << "\n"
                      << "   ↳ Stitched Input 0:    " << first_input << " (Direct camera data routing)\n"
                      << "   ↳ Fused Weight Scale:  " << first_step.weight_scale << "\n"
                      << "   ↳ Hardware Stride Jmp: " << first_step.custom_stride_step << " (Zero-Copy layout view)\n";
        }
        edge_engine.verify_compiled_memory_bounds(pruned_execution_graph);

        std::cout << "\n🚀 Engine is fully primed. System memory tracks are ready for kernel processing loops.\n";
        return 0;
        // tracy::SetThreadName("Main Processing Thread");
        // PRINT_INFO("============= Live Object Recognition Application =============");

        // abs_model_path = ResolveAbsModelPath(argv[0]);
        // PRINT_INFO("[ObjRec] Resolved Model Path : " << abs_model_path);

        // if (!pipeline.InitializeEngine(abs_model_path, intra_op_threads, inter_op_threads))
        // {
        //     PRINT_ERROR("[ObjRec] Failed to initialize ONNX Runtime engine.");
        //     return -1;
        // }
        // PRINT_INFO("[ObjRec] ONNX Engine initialized successfully.");

        // std::vector<std::string> model_classes = pipeline.GetClassNames();
        // if (model_classes.empty())
        // {
        //     PRINT_ERROR("[ObjRec] Could not resolve embedded model metadata strings. Falling back to numeric Class IDs.");
        // }
        // else
        // {
        //     PRINT_INFO("[ObjRec] Successfully parsed and initialized [" << model_classes.size() << "] distinct text class tags directly from the ONNX binary.");
        // }

        // if (!frame_handler.Initialize(std::string{kIphoneStreamUrl}))
        // {
        //     PRINT_ERROR("[ObjRec] Could not connect to live iPhone stream pipeline.");
        //     return -1;
        // }
        // frame_handler.Start();

        // PRINT_INFO("🚀 Starting Live iPhone Inference Loop.");

        // /* Hot-Path Inference Loop, no dynamic memory allocations after this point! */
        // bool application_active = true;
        // while (application_active)
        // {
        //     ZoneScopedN("Live Frame Loop");
        //     if (!frame_handler.FetchLatestFrame(live_frame))
        //     {
        //         if (!frame_handler.IsRunning())
        //         {
        //             PRINT_ERROR("[ObjRec] Camera stream disconnected. Stopping.");
        //             application_active = false;
        //             break;
        //         }
        //         if (cv::waitKey(1) == 'q')
        //         {
        //             PRINT_INFO("'q' key pressed while engine was idling. Stopping cleanly.");
        //             application_active = false;
        //             break;
        //         }
        //         continue; // No new frame available yet; jump back to check next loop slice
        //     }
        //     {
        //         ZoneScopedN("Preprocessing");
        //         processed_counter++;
        //         PRINT_INFO("-> Processing Live Camera Frame [" << processed_counter << "]");

        //         // Map incoming live frame buffers directly into structural signal processing dimensions
        //         frame_buffer.data_ptr = live_frame.data.data();
        //         frame_buffer.width = live_frame.width;
        //         frame_buffer.height = live_frame.height;
        //         frame_buffer.channels = live_frame.channels;
        //         frame_buffer.stride = live_frame.stride;

        //         if (!pipeline.StreamFramePreprocess(frame_buffer))
        //             continue;
        //     }
        //     {
        //         ZoneScopedN("Inference");
        //         if (!pipeline.InferenceExecute())
        //         {
        //             PRINT_ERROR("   [Inference Failure] Engine run failed for frame [" << processed_counter << "]");
        //             if (cv::waitKey(1) == 'q')
        //                 break;
        //             continue;
        //         }
        //     }

        //     /* Postprocessing to be moved to a separate thread/application */
        //     TransformMetadata meta = pipeline.GetLatestTransformMetadata();
        //     std::vector<Detection> detections;
        //     {
        //         ZoneScopedN("Postprocessing");
        //         detections = post_processor.Execute(
        //             out_preallocated_buffer.data(),
        //             0.75f, // Confidence Threshold
        //             0.45f, // NMS Overlap IoU Threshold
        //             meta   // Transform metrics containing source sizing bounds
        //         );
        //     }

        //     {
        //         ZoneScopedN("Visualization");
        //         // Initialize OpenCV container directly referencing the incoming camera frame array
        //         cv::Mat frame(live_frame.height, live_frame.width, CV_8UC3, live_frame.data.data(), live_frame.stride);
        //         cv::Mat visualization_frame = frame.clone();

        //         // Draw localized text strings and bounding boxes onto your cloned matrix canvas
        //         obj_rec::app::YOLOv8Visualizer::DrawDetections(visualization_frame, detections, model_classes);

        //         // Display your updated visualization frame
        //         cv::imshow("YOLOv8 Live iPhone Cam Canvas", visualization_frame);

        //         // Optional: Save latest processed artifact file directly onto disk
        //         cv::imwrite("output_visualized.jpg", visualization_frame);

        //         PRINT_INFO("   [Detections Filtered]: " << detections.size() << " objects tracked.");
        //         for (size_t i = 0; i < detections.size(); ++i)
        //         {
        //             std::string class_name = (detections[i].class_id >= 0 && static_cast<size_t>(detections[i].class_id) < model_classes.size())
        //                                          ? model_classes[detections[i].class_id]
        //                                          : "Unknown";

        //             PRINT_INFO("      Object [" << i + 1 << "] -> " << class_name
        //                                         << " | Conf: " << (detections[i].confidence * 100.0f) << "%"
        //                                         << " | Mapped Box: ["
        //                                         << detections[i].x1 << ", " << detections[i].y1 << ", "
        //                                         << detections[i].x2 << ", " << detections[i].y2 << "]");
        //         }
        //         PRINT_INFO("---------------------------------------------------------");
        //     }
        //     FrameMark;
        // }
        // frame_handler.Stop();
        // PRINT_INFO("🟩 Live Execution Terminated cleanly. Total frames processed: " << processed_counter);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] Critical Exception Aborted Operation: " << e.what() << "\n";
        return -1;
    }

    return 0;
}
