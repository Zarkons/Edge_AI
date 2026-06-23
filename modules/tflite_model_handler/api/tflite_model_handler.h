#ifndef TFLITE_MODEL_HANDLER_H_
#define TFLITE_MODEL_HANDLER_H_

#include <cstdint>
#include <functional>
#include <cstring>
#include <vector>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "common_types.h"

using namespace MlTypesDynamic;
using namespace QuantizationTypes;

template <int OpCount>
class TFLiteModelHandler
{
public:
    using ResolverType = tflite::MicroMutableOpResolver<OpCount>;
    using RegisterOpFn = std::function<void(ResolverType &)>;

    virtual ~TFLiteModelHandler() = default;

    // Public API methods that main.cc will call
    virtual bool Init(const unsigned char *model_data) = 0;
    // Overload 1: Returns raw, un-dequantized INT8 results (Blazing fast ArgMax style)
    virtual PredictionQuantInt8Vec Predict(const void *quantized_data, size_t bytes_to_copy) = 0;
    // Overload 2: Automatically de-quantizes the output into floats (Threshold style)
    virtual PredictionFloatVec Predict(const void *quantized_data, size_t bytes_to_copy, bool dequantize_output) = 0;
    virtual QuantizationParams GetInputQuantizationParams() const = 0;
    // Abstract registration hook to dynamically add ops from main
    virtual void RegisterOp(const RegisterOpFn &reg_func) = 0;
};

// Forward declaration of the factory creator function
template <size_t ArenaSize, int OpCount>
TFLiteModelHandler<OpCount> *CreateTFLiteModelHandler();

template <size_t ArenaSize, int OpCount>
class TFLiteModelHandlerImpl : public TFLiteModelHandler<OpCount>
{
private:
    using Base = TFLiteModelHandler<OpCount>;
    using ResolverType = typename Base::ResolverType;

    uint8_t tensor_arena_[ArenaSize];
    const tflite::Model *model_ = nullptr;
    tflite::MicroInterpreter *interpreter_ = nullptr;
    ResolverType op_resolver_;
    TfLiteTensor *input_tensor_ = nullptr;
    TfLiteTensor *output_tensor_ = nullptr;

public:
    TFLiteModelHandlerImpl() = default;

    ~TFLiteModelHandlerImpl() override
    {
        if (interpreter_ != nullptr)
        {
            delete interpreter_;
        }
    }

    void RegisterOp(const typename Base::RegisterOpFn &reg_func) override
    {
        reg_func(op_resolver_);
    }

    bool Init(const unsigned char *model_data) override
    {
        model_ = tflite::GetModel(model_data);
        if (model_->version() != TFLITE_SCHEMA_VERSION)
        {
            return false;
        }

        interpreter_ = new tflite::MicroInterpreter(model_, op_resolver_, tensor_arena_, ArenaSize);
        if (interpreter_->AllocateTensors() != kTfLiteOk)
        {
            return false;
        }

        input_tensor_ = interpreter_->input(0);
        output_tensor_ = interpreter_->output(0);
        return true;
    }

    PredictionQuantInt8Vec Predict(const void *quantized_data, size_t bytes_to_copy) override
    {
        if (bytes_to_copy > input_tensor_->bytes)
        {
            bytes_to_copy = input_tensor_->bytes;
        }
        std::memcpy(input_tensor_->data.raw, quantized_data, bytes_to_copy);

        if (interpreter_->Invoke() != kTfLiteOk)
        {
            return {};
        }

        // Grab raw bytes directly from the model output layer
        size_t output_elements = output_tensor_->bytes;
        PredictionQuantInt8Vec raw_predictions(output_elements);
        std::memcpy(raw_predictions.data(), output_tensor_->data.int8, output_elements);

        return raw_predictions;
    }

    PredictionFloatVec Predict(const void *quantized_data, size_t bytes_to_copy, bool dequantize_output) override
    {
        QuantizationParams output_quant_params{
            output_tensor_->params.scale,
            output_tensor_->params.zero_point};
        // Safety check to prevent memory corruption
        if (bytes_to_copy > input_tensor_->bytes)
        {
            bytes_to_copy = input_tensor_->bytes;
        }

        // Blindly copy the prepared bytes directly into the tensor data array
        std::memcpy(input_tensor_->data.raw, quantized_data, bytes_to_copy);

        if (interpreter_->Invoke() != kTfLiteOk)
        {
            return {};
        }

        // Process output dynamically based on tensor type (Float vs Int8)
        size_t output_elements = output_tensor_->bytes;
        PredictionFloatVec predictions;

        if (dequantize_output && output_tensor_->type == kTfLiteInt8)
        {
            predictions.resize(output_elements);
            for (size_t i = 0; i < output_elements; ++i)
            {
                predictions[i] = (output_tensor_->data.int8[i] - output_quant_params.zero_point) * output_quant_params.scale;
            }
        }
        else if (output_tensor_->type == kTfLiteFloat32)
        {
            predictions.resize(output_elements / sizeof(float));
            std::memcpy(predictions.data(), output_tensor_->data.f, output_tensor_->bytes);
        }

        // ToDo: Rework the model so the softmax is part of the graph and handled internally by the interpreter. For now, we apply softmax normalization here in C++ to ensure the output is a valid probability distribution.
        // Apply softmax normalization to the predictions
        float max_logit = *std::max_element(predictions.begin(), predictions.end());
        float sum_exp = 0.0f;
        for (size_t i = 0; i < output_elements; ++i)
        {
            predictions[i] = std::exp(predictions[i] - max_logit); // Subtracting max_logit protects your stack
            sum_exp += predictions[i];
        }

        for (size_t i = 0; i < output_elements; ++i)
        {
            predictions[i] /= sum_exp;
        }

        return predictions;
    }

    QuantizationParams GetInputQuantizationParams() const override
    {
        QuantizationParams params;
        if (input_tensor_ != nullptr)
        {
            params.scale = input_tensor_->params.scale;
            params.zero_point = input_tensor_->params.zero_point;
        }
        return params;
    }
};

template <size_t ArenaSize, int OpCount>
TFLiteModelHandler<OpCount> *CreateTFLiteModelHandler()
{
    return new TFLiteModelHandlerImpl<ArenaSize, OpCount>();
}

#endif // TFLITE_MODEL_HANDLER_H_
