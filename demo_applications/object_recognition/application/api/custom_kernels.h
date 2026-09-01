#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "inference_pipeline_structures.h"

class CustomKernels
{
public:
    void execute(const RuntimeNode &node,
                 const std::vector<const void *> &input_lanes,
                 void *act_out) const;

    void execute_custom_conv(const void *act_in,
                             const int8_t *weights,
                             const int32_t *bias,
                             void *act_out,
                             const ExecutionStep &step,
                             const std::vector<int64_t> &input_shape,
                             const std::vector<int64_t> &output_shape) const;

    void execute_custom_sigmoid(const void *act_in,
                                void *act_out,
                                const ExecutionStep &step,
                                const std::vector<int64_t> &output_shape) const;

    void execute_custom_mul(const void *lhs,
                            const void *rhs,
                            void *act_out,
                            const ExecutionStep &step,
                            const std::vector<int64_t> &output_shape) const;

    void execute_custom_add(const void *lhs,
                            const void *rhs,
                            void *act_out,
                            const ExecutionStep &step,
                            const std::vector<int64_t> &output_shape) const;

    void execute_custom_concat(const std::vector<const void *> &input_lanes,
                               void *act_out,
                               const ExecutionStep &step,
                               const std::vector<std::vector<int64_t>> &input_shapes,
                               const std::vector<int64_t> &output_shape) const;

    void execute_custom_maxpool(const void *act_in,
                                void *act_out,
                                const ExecutionStep &step,
                                const std::vector<int64_t> &input_shape,
                                const std::vector<int64_t> &output_shape) const;

    void execute_custom_resize(const void *act_in,
                               void *act_out,
                               const ExecutionStep &step,
                               const std::vector<int64_t> &input_shape,
                               const std::vector<int64_t> &output_shape) const;

    void execute_custom_sub(const void *lhs,
                            const void *rhs,
                            void *act_out,
                            const ExecutionStep &step,
                            const std::vector<int64_t> &output_shape) const;

    void execute_custom_softmax(const void *act_in,
                                void *act_out,
                                const ExecutionStep &step,
                                const std::vector<int64_t> &output_shape) const;

    void execute_custom_div(const void *lhs,
                            const void *rhs,
                            void *act_out,
                            const ExecutionStep &step,
                            const std::vector<int64_t> &output_shape) const;
};
