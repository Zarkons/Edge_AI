#ifndef RTLITE_MICRO_HANDLER_H_
#define RTLITE_MICRO_HANDLER_H_

#include <cstdint>
#include <functional>
#include <cstring>
#include <vector>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "common_types.h"

using namespace ml_types_dynamic;
using namespace quantization_types;

namespace rtlite_micro_handler
{

    /**
     * @brief Generic handler interface for executing TensorFlow Lite Micro models.
     *
     * This class abstracts away the low-level memory handling, interpreter setup,
     * and tensor allocation required by TensorFlow Lite Micro.
     *
     * @tparam OpCount The total number of operations (layers) to reserve inside the mutable operator resolver.
     */
    template <int OpCount>
    class TFLiteModelHandler
    {
    public:
        /**
         * @brief Type alias for the TensorFlow Lite Micro mutable operator resolver.
         */
        using ResolverType = tflite::MicroMutableOpResolver<OpCount>;
        /**
         * @brief Type alias for the function used to register operations with the resolver.
         */
        using RegisterOpFn = std::function<void(ResolverType &)>;

        virtual ~TFLiteModelHandler() = default;

        /**
         * @brief Initializes the TensorFlow Lite Micro model handler with the provided model data.
         *
         * @param model_data Pointer to the raw model data (flatbuffer).
         * @return true if initialization is successful, false otherwise.
         */
        virtual bool Init(const unsigned char *model_data) = 0;
        /**
         * @brief Performs inference on the provided quantized input data.
         *
         * @param quantized_data Pointer to the quantized input data.
         * @param bytes_to_copy Number of bytes to copy from the input data.
         * @return Prediction results as a vector of INT8 values.
         */
        virtual PredictionQuantInt8Vec Predict(const void *quantized_data, size_t bytes_to_copy) = 0;
        /**
         * @brief Performs inference on the provided quantized input data and optionally de-quantizes the output.
         *
         * @param quantized_data Pointer to the quantized input data.
         * @param bytes_to_copy Number of bytes to copy from the input data.
         * @param dequantize_output Whether to de-quantize the output into floats.
         * @return Prediction results as a vector of floats.
         */
        virtual PredictionFloatVec Predict(const void *quantized_data, size_t bytes_to_copy, bool dequantize_output) = 0;
        /**
         * @brief Retrieves the quantization parameters for the input tensor.
         *
         * @return Quantization parameters for the input tensor (scale and zero-point).
         */
        virtual QuantizationParams GetInputQuantizationParams() const = 0;
        /**
         * @brief Abstract registration hook to dynamically add operations from the main function.
         *
         * @param reg_func Function used to register operations with the resolver.
         */
        virtual void RegisterOp(const RegisterOpFn &reg_func) = 0;
    };

    /**
     * @brief Factory function to create a TFLiteModelHandler instance with a specified tensor arena size and operation count.
     *
     * @tparam ArenaSize The size of the tensor arena in bytes.
     * @tparam OpCount The total number of operations (layers) to reserve inside the mutable operator resolver.
     * @return A pointer to the newly allocated @ref TFLiteModelHandler instance.
     */
    template <size_t ArenaSize, int OpCount>
    TFLiteModelHandler<OpCount> *CreateTFLiteModelHandler();

    /**
     * @brief Implementation of the TFLiteModelHandler interface for TensorFlow Lite Micro.
     *
     * This class manages the TensorFlow Lite Micro interpreter, tensor arena, and operation resolver.
     *
     * @tparam ArenaSize The size of the tensor arena in bytes.
     * @tparam OpCount The total number of operations (layers) to reserve inside the mutable operator resolver.
     */
    template <size_t ArenaSize, int OpCount>
    class TFLiteModelHandlerImpl : public TFLiteModelHandler<OpCount>
    {
    private:
        /**
         * @brief Type alias for the base TFLiteModelHandler class.
         */
        using Base = TFLiteModelHandler<OpCount>;
        /**
         * @brief Type alias for the TensorFlow Lite Micro mutable operator resolver.
         */
        using ResolverType = typename Base::ResolverType;

        /**
         * @brief Tensor arena for the TensorFlow Lite Micro interpreter.
         */
        uint8_t tensor_arena_[ArenaSize];
        /**
         * @brief Pointer to the TensorFlow Lite Micro model.
         */
        const tflite::Model *model_ = nullptr;
        /**
         * @brief Pointer to the TensorFlow Lite Micro interpreter.
         */
        tflite::MicroInterpreter *interpreter_ = nullptr;
        /**
         * @brief Mutable operator resolver for TensorFlow Lite Micro.
         */
        ResolverType op_resolver_;
        /**
         * @brief Pointer to the input tensor of the model.
         */
        TfLiteTensor *input_tensor_ = nullptr;
        /**
         * @brief Pointer to the output tensor of the model.
         */
        TfLiteTensor *output_tensor_ = nullptr;

    public:
        TFLiteModelHandlerImpl() = default;

        /**
         * @brief Destructor for the TFLiteModelHandlerImpl class.
         *
         * Cleans up the TensorFlow Lite Micro interpreter if it was created.
         */
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

        /**
         * @copydoc TFLiteModelHandler::Init
         *
         * @note Initialization is done by:
         *         1. Loading the model from the provided flatbuffer data.
         *         2. Instantiating the TensorFlow Lite Micro interpreter with the model, operator resolver, and tensor     arena.
         *         3. Allocating tensors for the model.
         *         4. Storing pointers to the input and output tensors for later use.
         */
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

        /**
         * @copydoc TFLiteModelHandler::Predict(const void *, size_t)
         *
         * @note This method performs inference by:
         *         1. Copying the provided quantized input data into the input tensor.
         *         2. Invoking the TensorFlow Lite Micro interpreter to run inference.
         *         3. Copying the raw output bytes from the output tensor into a vector of INT8 values.
         */
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

        /**
         * @copydoc TFLiteModelHandler::Predict(const void *, size_t, bool)
         *
         * @note This method performs inference by:
         *         1. Copying the provided quantized input data into the input tensor.
         *         2. Invoking the TensorFlow Lite Micro interpreter to run inference.
         *         3. Dequantizing the output data if requested.
         *         4. Applying softmax normalization to the predictions.
         */
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
            size_t output_elements = (output_tensor_->type == kTfLiteFloat32)
                                         ? (output_tensor_->bytes / sizeof(float))
                                         : output_tensor_->bytes;
            PredictionFloatVec predictions;

            predictions.resize(output_elements);
            if (dequantize_output && output_tensor_->type == kTfLiteInt8)
            {
                for (size_t i = 0; i < output_elements; ++i)
                {
                    predictions[i] = (output_tensor_->data.int8[i] - output_quant_params.zero_point) * output_quant_params.scale;
                }
            }
            else if (output_tensor_->type == kTfLiteFloat32)
            {
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

} // namespace rtlite_micro_handler

#endif // RTLITE_MICRO_HANDLER_H_
