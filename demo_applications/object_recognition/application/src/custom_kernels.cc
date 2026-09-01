#include "custom_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace
{
    struct Shape4D
    {
        int64_t n = 0;
        int64_t c = 0;
        int64_t h = 0;
        int64_t w = 0;
    };

    int64_t get_or_default(const std::vector<int64_t> &values, size_t idx, int64_t fallback)
    {
        return (idx < values.size()) ? values[idx] : fallback;
    }

    int64_t clamp_non_negative(int64_t v)
    {
        return (v < 0) ? 0 : v;
    }

    bool to_shape4d(const std::vector<int64_t> &shape, Shape4D &out)
    {
        if (shape.size() != 4)
        {
            return false;
        }

        out.n = shape[0];
        out.c = shape[1];
        out.h = shape[2];
        out.w = shape[3];
        return (out.n > 0 && out.c > 0 && out.h > 0 && out.w > 0);
    }

    size_t nchw_index(const Shape4D &s, int64_t n, int64_t c, int64_t h, int64_t w)
    {
        return static_cast<size_t>(((n * s.c + c) * s.h + h) * s.w + w);
    }

    int64_t normalize_axis(int64_t axis, int64_t rank)
    {
        return (axis < 0) ? (axis + rank) : axis;
    }

    size_t axis_inner_stride(const std::vector<int64_t> &shape, int64_t axis)
    {
        size_t stride = 1;
        for (size_t i = static_cast<size_t>(axis + 1); i < shape.size(); ++i)
        {
            stride *= static_cast<size_t>(shape[i]);
        }
        return stride;
    }

    size_t axis_outer_count(const std::vector<int64_t> &shape, int64_t axis)
    {
        size_t outer = 1;
        for (size_t i = 0; i < static_cast<size_t>(axis); ++i)
        {
            outer *= static_cast<size_t>(shape[i]);
        }
        return outer;
    }

    size_t compute_tensor_element_count(const std::vector<int64_t> &shape)
    {
        if (shape.empty())
        {
            return 0;
        }

        size_t elements = 1;
        for (const int64_t dim : shape)
        {
            if (dim <= 0)
            {
                return 0;
            }
            elements *= static_cast<size_t>(dim);
        }
        return elements;
    }

    size_t compute_tensor_byte_count(const std::vector<int64_t> &shape)
    {
        return compute_tensor_element_count(shape) * sizeof(float);
    }

    void copy_tensor_if_valid(const void *src, void *dst, const std::vector<int64_t> &shape)
    {
        if (!src || !dst)
        {
            return;
        }

        const size_t bytes = compute_tensor_byte_count(shape);
        if (bytes == 0)
        {
            return;
        }

        if (src != dst)
        {
            std::memcpy(dst, src, bytes);
        }
    }

    float safe_divide(float lhs, float rhs)
    {
        return (std::fabs(rhs) <= 1e-12f) ? 0.0f : (lhs / rhs);
    }
} // namespace

void CustomKernels::execute(const RuntimeNode &node,
                            const std::vector<const void *> &input_lanes,
                            void *act_out) const
{
    const std::vector<int64_t> &output_shape = node.output_shape;

    const void *primary_act_in = nullptr;
    const void *secondary_act_in = nullptr;

    for (const void *candidate : input_lanes)
    {
        if (candidate == nullptr)
        {
            continue;
        }

        if (primary_act_in == nullptr)
        {
            primary_act_in = candidate;
            continue;
        }

        secondary_act_in = candidate;
        break;
    }

    std::vector<int64_t> fallback_input_shape = output_shape;
    const std::vector<int64_t> *primary_input_shape = &fallback_input_shape;

    if (node.primary_shape_idx < node.input_shapes.size() && !node.input_shapes[node.primary_shape_idx].empty())
    {
        primary_input_shape = &node.input_shapes[node.primary_shape_idx];
    }
    else
    {
        for (const auto &shape : node.input_shapes)
        {
            if (!shape.empty())
            {
                primary_input_shape = &shape;
                break;
            }
        }
    }

    if (node.op_type == "Conv")
    {
        execute_custom_conv(primary_act_in,
                            node.blueprint.weights_ptr,
                            node.blueprint.bias_ptr,
                            act_out,
                            node.blueprint,
                            *primary_input_shape,
                            output_shape);
    }
    else if (node.op_type == "Sigmoid")
    {
        execute_custom_sigmoid(primary_act_in, act_out, node.blueprint, output_shape);
    }
    else if (node.op_type == "Mul")
    {
        execute_custom_mul(primary_act_in, secondary_act_in, act_out, node.blueprint, output_shape);
    }
    else if (node.op_type == "Add")
    {
        execute_custom_add(primary_act_in, secondary_act_in, act_out, node.blueprint, output_shape);
    }
    else if (node.op_type == "Concat")
    {
        execute_custom_concat(input_lanes, act_out, node.blueprint, node.input_shapes, output_shape);
    }
    else if (node.op_type == "MaxPool")
    {
        execute_custom_maxpool(primary_act_in, act_out, node.blueprint, *primary_input_shape, output_shape);
    }
    else if (node.op_type == "Resize")
    {
        execute_custom_resize(primary_act_in, act_out, node.blueprint, *primary_input_shape, output_shape);
    }
    else if (node.op_type == "Sub")
    {
        execute_custom_sub(primary_act_in, secondary_act_in, act_out, node.blueprint, output_shape);
    }
    else if (node.op_type == "Softmax")
    {
        execute_custom_softmax(primary_act_in, act_out, node.blueprint, output_shape);
    }
    else if (node.op_type == "Div")
    {
        execute_custom_div(primary_act_in, secondary_act_in, act_out, node.blueprint, output_shape);
    }
    else
    {
        copy_tensor_if_valid(primary_act_in, act_out, output_shape);
    }
}

void CustomKernels::execute_custom_conv(const void *act_in,
                                        const int8_t *weights,
                                        const int32_t *bias,
                                        void *act_out,
                                        const ExecutionStep &step,
                                        const std::vector<int64_t> &input_shape,
                                        const std::vector<int64_t> &output_shape) const
{
    if (!act_in || !weights || !act_out)
    {
        return;
    }

    Shape4D out_shape;
    if (!to_shape4d(output_shape, out_shape))
    {
        copy_tensor_if_valid(act_in, act_out, output_shape);
        return;
    }

    Shape4D in_shape;
    if (!to_shape4d(input_shape, in_shape))
    {
        copy_tensor_if_valid(act_in, act_out, output_shape);
        return;
    }

    const auto *in = reinterpret_cast<const float *>(act_in);
    auto *out = reinterpret_cast<float *>(act_out);

    const int64_t groups = std::max<int64_t>(1, step.group);
    const int64_t oc_per_group = out_shape.c / groups;
    const int64_t ic_per_group = std::max<int64_t>(1, in_shape.c / groups);

    const int64_t stride_h = std::max<int64_t>(1, get_or_default(step.strides, 0, 1));
    const int64_t stride_w = std::max<int64_t>(1, get_or_default(step.strides, 1, stride_h));

    const int64_t pad_top = get_or_default(step.pads, 0, 0);
    const int64_t pad_left = get_or_default(step.pads, 1, 0);

    const int64_t dilation_h = std::max<int64_t>(1, get_or_default(step.dilations, 0, 1));
    const int64_t dilation_w = std::max<int64_t>(1, get_or_default(step.dilations, 1, dilation_h));

    const int64_t k_h = (step.kernel_shape.size() >= 2)
                            ? std::max<int64_t>(1, step.kernel_shape[step.kernel_shape.size() - 2])
                            : 3;
    const int64_t k_w = (step.kernel_shape.size() >= 1)
                            ? std::max<int64_t>(1, step.kernel_shape[step.kernel_shape.size() - 1])
                            : 3;

    const float input_scale = (step.input_scale > 0.0f) ? step.input_scale : 1.0f;
    const float weight_scale = (step.weight_scale > 0.0f) ? step.weight_scale : 1.0f;
    const float bias_scale = input_scale * weight_scale;
    const int32_t weight_zp = step.weight_zero_point;

    for (int64_t n = 0; n < out_shape.n; ++n)
    {
        for (int64_t oc = 0; oc < out_shape.c; ++oc)
        {
            const int64_t group_idx = (oc_per_group > 0) ? (oc / oc_per_group) : 0;
            for (int64_t oh = 0; oh < out_shape.h; ++oh)
            {
                for (int64_t ow = 0; ow < out_shape.w; ++ow)
                {
                    float acc = 0.0f;
                    if (bias != nullptr)
                    {
                        acc = static_cast<float>(bias[oc]) * bias_scale;
                    }

                    for (int64_t ic_local = 0; ic_local < ic_per_group; ++ic_local)
                    {
                        const int64_t ic = group_idx * ic_per_group + ic_local;
                        for (int64_t kh = 0; kh < k_h; ++kh)
                        {
                            for (int64_t kw = 0; kw < k_w; ++kw)
                            {
                                const int64_t ih = oh * stride_h - pad_top + kh * dilation_h;
                                const int64_t iw = ow * stride_w - pad_left + kw * dilation_w;

                                if (ih < 0 || ih >= in_shape.h || iw < 0 || iw >= in_shape.w)
                                {
                                    continue;
                                }

                                const size_t in_idx = nchw_index(in_shape, n, ic, ih, iw);
                                const size_t w_idx = static_cast<size_t>((((oc * ic_per_group + ic_local) * k_h) + kh) * k_w + kw);

                                const float x = in[in_idx];
                                const float w = (static_cast<float>(weights[w_idx]) - static_cast<float>(weight_zp)) * weight_scale;
                                acc += x * w;
                            }
                        }
                    }

                    const size_t out_idx = nchw_index(out_shape, n, oc, oh, ow);
                    out[out_idx] = acc;
                }
            }
        }
    }
}

void CustomKernels::execute_custom_sigmoid(const void *act_in,
                                           void *act_out,
                                           const ExecutionStep &step,
                                           const std::vector<int64_t> &output_shape) const
{
    (void)step;
    if (!act_in || !act_out)
    {
        return;
    }

    const size_t elements = compute_tensor_element_count(output_shape);
    const auto *in = reinterpret_cast<const float *>(act_in);
    auto *out = reinterpret_cast<float *>(act_out);

    for (size_t i = 0; i < elements; ++i)
    {
        out[i] = 1.0f / (1.0f + std::exp(-in[i]));
    }
}

void CustomKernels::execute_custom_mul(const void *lhs,
                                       const void *rhs,
                                       void *act_out,
                                       const ExecutionStep &step,
                                       const std::vector<int64_t> &output_shape) const
{
    (void)step;
    if (!lhs || !rhs || !act_out)
    {
        return;
    }

    const size_t elements = compute_tensor_element_count(output_shape);
    const auto *in_lhs = reinterpret_cast<const float *>(lhs);
    const auto *in_rhs = reinterpret_cast<const float *>(rhs);
    auto *out = reinterpret_cast<float *>(act_out);

    for (size_t i = 0; i < elements; ++i)
    {
        out[i] = in_lhs[i] * in_rhs[i];
    }
}

void CustomKernels::execute_custom_add(const void *lhs,
                                       const void *rhs,
                                       void *act_out,
                                       const ExecutionStep &step,
                                       const std::vector<int64_t> &output_shape) const
{
    (void)step;
    if (!lhs || !rhs || !act_out)
    {
        return;
    }

    const size_t elements = compute_tensor_element_count(output_shape);
    const auto *in_lhs = reinterpret_cast<const float *>(lhs);
    const auto *in_rhs = reinterpret_cast<const float *>(rhs);
    auto *out = reinterpret_cast<float *>(act_out);

    for (size_t i = 0; i < elements; ++i)
    {
        out[i] = in_lhs[i] + in_rhs[i];
    }
}

void CustomKernels::execute_custom_concat(const std::vector<const void *> &input_lanes,
                                          void *act_out,
                                          const ExecutionStep &step,
                                          const std::vector<std::vector<int64_t>> &input_shapes,
                                          const std::vector<int64_t> &output_shape) const
{
    if (!act_out || output_shape.empty())
    {
        return;
    }

    const int64_t rank = static_cast<int64_t>(output_shape.size());
    const int64_t axis = normalize_axis(step.axis, rank);
    if (axis < 0 || axis >= rank)
    {
        return;
    }

    const size_t outer = axis_outer_count(output_shape, axis);
    const size_t inner = axis_inner_stride(output_shape, axis);
    const size_t out_axis_len = static_cast<size_t>(output_shape[axis]);
    auto *out = reinterpret_cast<float *>(act_out);

    size_t axis_offset = 0;
    const size_t lane_count = std::min(input_lanes.size(), input_shapes.size());
    for (size_t lane_idx = 0; lane_idx < lane_count; ++lane_idx)
    {
        const void *lane_ptr = input_lanes[lane_idx];
        if (lane_ptr == nullptr)
        {
            continue;
        }

        const std::vector<int64_t> &in_shape = input_shapes[lane_idx];
        if (in_shape.size() != output_shape.size())
        {
            continue;
        }

        const size_t in_axis_len = static_cast<size_t>(in_shape[axis]);
        if (in_axis_len == 0)
        {
            continue;
        }

        const auto *in = reinterpret_cast<const float *>(lane_ptr);
        const size_t copy_axis_len = std::min(in_axis_len, out_axis_len - axis_offset);

        for (size_t o = 0; o < outer; ++o)
        {
            const size_t src_base = o * in_axis_len * inner;
            const size_t dst_base = o * out_axis_len * inner + axis_offset * inner;
            std::memcpy(out + dst_base, in + src_base, copy_axis_len * inner * sizeof(float));
        }

        axis_offset += copy_axis_len;
        if (axis_offset >= out_axis_len)
        {
            break;
        }
    }
}

void CustomKernels::execute_custom_maxpool(const void *act_in,
                                           void *act_out,
                                           const ExecutionStep &step,
                                           const std::vector<int64_t> &input_shape,
                                           const std::vector<int64_t> &output_shape) const
{
    if (!act_in || !act_out)
    {
        return;
    }

    Shape4D out_shape;
    if (!to_shape4d(output_shape, out_shape))
    {
        copy_tensor_if_valid(act_in, act_out, output_shape);
        return;
    }

    Shape4D in_shape;
    if (!to_shape4d(input_shape, in_shape))
    {
        copy_tensor_if_valid(act_in, act_out, output_shape);
        return;
    }

    const auto *in = reinterpret_cast<const float *>(act_in);
    auto *out = reinterpret_cast<float *>(act_out);

    const int64_t stride_h = std::max<int64_t>(1, get_or_default(step.strides, 0, 1));
    const int64_t stride_w = std::max<int64_t>(1, get_or_default(step.strides, 1, stride_h));

    const int64_t pad_top = get_or_default(step.pads, 0, 0);
    const int64_t pad_left = get_or_default(step.pads, 1, 0);

    const int64_t dilation_h = std::max<int64_t>(1, get_or_default(step.dilations, 0, 1));
    const int64_t dilation_w = std::max<int64_t>(1, get_or_default(step.dilations, 1, dilation_h));

    const int64_t k_h = std::max<int64_t>(1, get_or_default(step.kernel_shape, 0, 2));
    const int64_t k_w = std::max<int64_t>(1, get_or_default(step.kernel_shape, 1, k_h));

    for (int64_t n = 0; n < out_shape.n; ++n)
    {
        for (int64_t c = 0; c < out_shape.c; ++c)
        {
            for (int64_t oh = 0; oh < out_shape.h; ++oh)
            {
                for (int64_t ow = 0; ow < out_shape.w; ++ow)
                {
                    float max_v = -std::numeric_limits<float>::infinity();
                    bool found = false;

                    for (int64_t kh = 0; kh < k_h; ++kh)
                    {
                        for (int64_t kw = 0; kw < k_w; ++kw)
                        {
                            const int64_t ih = oh * stride_h - pad_top + kh * dilation_h;
                            const int64_t iw = ow * stride_w - pad_left + kw * dilation_w;

                            if (ih < 0 || ih >= in_shape.h || iw < 0 || iw >= in_shape.w)
                            {
                                continue;
                            }

                            const size_t in_idx = nchw_index(in_shape, n, c, ih, iw);
                            max_v = found ? std::max(max_v, in[in_idx]) : in[in_idx];
                            found = true;
                        }
                    }

                    const size_t out_idx = nchw_index(out_shape, n, c, oh, ow);
                    out[out_idx] = found ? max_v : 0.0f;
                }
            }
        }
    }
}

void CustomKernels::execute_custom_resize(const void *act_in,
                                          void *act_out,
                                          const ExecutionStep &step,
                                          const std::vector<int64_t> &input_shape,
                                          const std::vector<int64_t> &output_shape) const
{
    (void)step;
    if (!act_in || !act_out)
    {
        return;
    }

    Shape4D out_shape;
    if (!to_shape4d(output_shape, out_shape))
    {
        copy_tensor_if_valid(act_in, act_out, output_shape);
        return;
    }

    Shape4D in_shape;
    if (!to_shape4d(input_shape, in_shape))
    {
        copy_tensor_if_valid(act_in, act_out, output_shape);
        return;
    }

    const auto *in = reinterpret_cast<const float *>(act_in);
    auto *out = reinterpret_cast<float *>(act_out);

    const float scale_h = static_cast<float>(in_shape.h) / static_cast<float>(out_shape.h);
    const float scale_w = static_cast<float>(in_shape.w) / static_cast<float>(out_shape.w);

    for (int64_t n = 0; n < out_shape.n; ++n)
    {
        for (int64_t c = 0; c < out_shape.c; ++c)
        {
            for (int64_t oh = 0; oh < out_shape.h; ++oh)
            {
                const int64_t ih = std::min<int64_t>(in_shape.h - 1, clamp_non_negative(static_cast<int64_t>(std::floor(oh * scale_h))));
                for (int64_t ow = 0; ow < out_shape.w; ++ow)
                {
                    const int64_t iw = std::min<int64_t>(in_shape.w - 1, clamp_non_negative(static_cast<int64_t>(std::floor(ow * scale_w))));
                    const size_t in_idx = nchw_index(in_shape, n, c, ih, iw);
                    const size_t out_idx = nchw_index(out_shape, n, c, oh, ow);
                    out[out_idx] = in[in_idx];
                }
            }
        }
    }
}

void CustomKernels::execute_custom_sub(const void *lhs,
                                       const void *rhs,
                                       void *act_out,
                                       const ExecutionStep &step,
                                       const std::vector<int64_t> &output_shape) const
{
    (void)step;
    if (!lhs || !rhs || !act_out)
    {
        return;
    }

    const size_t elements = compute_tensor_element_count(output_shape);
    const auto *in_lhs = reinterpret_cast<const float *>(lhs);
    const auto *in_rhs = reinterpret_cast<const float *>(rhs);
    auto *out = reinterpret_cast<float *>(act_out);

    for (size_t i = 0; i < elements; ++i)
    {
        out[i] = in_lhs[i] - in_rhs[i];
    }
}

void CustomKernels::execute_custom_softmax(const void *act_in,
                                           void *act_out,
                                           const ExecutionStep &step,
                                           const std::vector<int64_t> &output_shape) const
{
    if (!act_in || !act_out)
    {
        return;
    }

    const size_t elements = compute_tensor_element_count(output_shape);
    if (elements == 0)
    {
        return;
    }

    const auto *in = reinterpret_cast<const float *>(act_in);
    auto *out = reinterpret_cast<float *>(act_out);

    const int64_t rank = static_cast<int64_t>(output_shape.size());
    const int64_t axis = normalize_axis(step.axis, rank);
    if (axis < 0 || axis >= rank)
    {
        copy_tensor_if_valid(act_in, act_out, output_shape);
        return;
    }

    const size_t axis_len = static_cast<size_t>(output_shape[axis]);
    if (axis_len == 0)
    {
        return;
    }

    const size_t inner = axis_inner_stride(output_shape, axis);
    const size_t outer = axis_outer_count(output_shape, axis);

    for (size_t o = 0; o < outer; ++o)
    {
        for (size_t i = 0; i < inner; ++i)
        {
            const size_t base = o * axis_len * inner + i;

            float max_value = -std::numeric_limits<float>::infinity();
            for (size_t a = 0; a < axis_len; ++a)
            {
                max_value = std::max(max_value, in[base + a * inner]);
            }

            float exp_sum = 0.0f;
            for (size_t a = 0; a < axis_len; ++a)
            {
                const float e = std::exp(in[base + a * inner] - max_value);
                out[base + a * inner] = e;
                exp_sum += e;
            }

            if (exp_sum > 0.0f)
            {
                for (size_t a = 0; a < axis_len; ++a)
                {
                    out[base + a * inner] /= exp_sum;
                }
            }
        }
    }
}

void CustomKernels::execute_custom_div(const void *lhs,
                                       const void *rhs,
                                       void *act_out,
                                       const ExecutionStep &step,
                                       const std::vector<int64_t> &output_shape) const
{
    (void)step;
    if (!lhs || !rhs || !act_out)
    {
        return;
    }

    const size_t elements = compute_tensor_element_count(output_shape);
    const auto *in_lhs = reinterpret_cast<const float *>(lhs);
    const auto *in_rhs = reinterpret_cast<const float *>(rhs);
    auto *out = reinterpret_cast<float *>(act_out);

    for (size_t i = 0; i < elements; ++i)
    {
        out[i] = safe_divide(in_lhs[i], in_rhs[i]);
    }
}
