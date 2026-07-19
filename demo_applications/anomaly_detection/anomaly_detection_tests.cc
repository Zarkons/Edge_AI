/* Test includes */
#include "gtest/gtest.h"

/* Standard library includes */
#include <random>
#include <vector>
#include <iostream>
#include <algorithm>

/* User defined includes */
#include "anomaly_detection_quantized.h"

/* TensorFlow Lite includes */
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

#define TENSOR_ARENA_SIZE 2 * 1024
/* Threshold set based on training data from the model */
const float ANOMALY_THRESHOLD = 0.013609121553599834f;
void FillNormalInput(float *output_buffer, size_t size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> distribution(0.5f, 0.05f);

    for (size_t i = 0; i < size; ++i)
    {
        output_buffer[i] = distribution(gen); // Writes straight into the Tensor Arena
    }
}

void FillAnomalousInput(float *output_buffer, size_t size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> distribution(5.0f, 0.05f); // Shifted mean to create anomaly

    for (size_t i = 0; i < size; ++i)
    {
        output_buffer[i] = distribution(gen); // Writes straight into the Tensor Arena
    }
}

void PrintTensorData(const float *input_data, size_t size)
{
    std::cout << "Tensor Data: ";
    for (size_t i = 0; i < size; ++i)
    {
        std::cout << input_data[i] << " ";
    }
    std::cout << std::endl;
}

float CalculateMSE(const float *predictions, const float *ground_truth, size_t size)
{
    float mse = 0.0f;
    for (size_t i = 0; i < size; ++i)
    {
        float error = predictions[i] - ground_truth[i];
        mse += error * error;
    }
    return mse / static_cast<float>(size);
}

TEST(AnomalyDetectionTest, TestInference)
{
    std::cout << "Running Anomaly Detection Test" << std::endl;
    const tflite::Model *model = ::tflite::GetModel(anomaly_detection_quantized_tflite);

    ASSERT_EQ(model->version(), TFLITE_SCHEMA_VERSION) << "Model schema version mismatch!";

    /* Create an operator resolver that will include all required operations
    This way only necessary operations are baked into the final binary, saving space.
    To achieve this, some linker optimization is required */
    tflite::MicroMutableOpResolver<4> op_resolver;
    ASSERT_EQ(op_resolver.AddFullyConnected(), kTfLiteOk);
    ASSERT_EQ(op_resolver.AddRelu(), kTfLiteOk);
    ASSERT_EQ(op_resolver.AddQuantize(), kTfLiteOk);
    ASSERT_EQ(op_resolver.AddDequantize(), kTfLiteOk);

    /* Create the interpreter that will run inference on the model based on inputs */
    uint8_t tensor_arena[TENSOR_ARENA_SIZE];
    tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena, TENSOR_ARENA_SIZE);
    ASSERT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

    /* Interpreter get the pointer to input tensor from the allocated tensor arena  */
    TfLiteTensor *input = interpreter.input(0);
    ASSERT_NE(input, nullptr);
    /* Even though the input is a 1D array, the tensor has 2 dimensions(1, 3) */
    ASSERT_EQ(2, input->dims->size);
    ASSERT_EQ(1, input->dims->data[0]);
    ASSERT_EQ(3, input->dims->data[1]);
    ASSERT_EQ(kTfLiteFloat32, input->type);

    float *input_data = tflite::GetTensorData<float>(input);
    // FillAnomalousInput(input_data, 3);
    FillNormalInput(input_data, 3);
    PrintTensorData(input_data, 3);
    float saved_input_data[3];
    std::copy(input_data, input_data + 3, saved_input_data);

    TfLiteStatus invoke_status = interpreter.Invoke();
    ASSERT_EQ(invoke_status, kTfLiteOk);

    /* Get the output tensor */
    TfLiteTensor *output = interpreter.output(0);
    ASSERT_NE(output, nullptr);
    ASSERT_EQ(2, output->dims->size);
    ASSERT_EQ(1, output->dims->data[0]);
    ASSERT_EQ(3, output->dims->data[1]);
    ASSERT_EQ(kTfLiteFloat32, output->type);
    const float *output_data = tflite::GetTensorData<float>(output);
    PrintTensorData(output_data, 3);
    PrintTensorData(input_data, 3);

    /* Calculate MSE for the output */
    float mse = CalculateMSE(output_data, saved_input_data, 3);
    std::cout << "Calculated MSE: " << mse << std::endl;
    if (mse > ANOMALY_THRESHOLD)
    {
        std::cout << "Anomaly detected! MSE: " << mse << std::endl;
    }
    else
    {
        std::cout << "No anomaly detected. MSE: " << mse << std::endl;
    }

    /* Check if the output is below the threshold */
    // float output_value = tflite::GetTensorData<float>(output)[0];
}