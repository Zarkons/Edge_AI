load("@pip_deps_tfcore//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary")

def _tflite_inspect_model_macro_impl(name, model_target, visibility = None):
    py_binary(
        name = name,
        srcs = ["//modules/helpers:tflite_inspect_model.py"],
        main = "//modules/helpers:tflite_inspect_model.py",
        args = [
            "--model_path=$(rootpath {})".format(model_target),
        ],
        data = [model_target],
        visibility = visibility,
        deps = [
            "@rules_python//python/runfiles",
        ],
    )

tflite_inspect_model_macro = macro(
    implementation = _tflite_inspect_model_macro_impl,
    attrs = {
        "model_target": attr.label(
            mandatory = True,
            configurable = False,
        ),
    },
)

def _onnxruntime_inspect_model_macro_impl(name, model_target, visibility = None):
    py_binary(
        name = name,
        srcs = ["//modules/helpers:onnxruntime_inspect_model.py"],
        main = "//modules/helpers:onnxruntime_inspect_model.py",
        args = [
            "--model_path=$(rootpath {})".format(model_target),
        ],
        data = [model_target],
        visibility = visibility,
        deps = [
            "@rules_python//python/runfiles",
            requirement("onnx"),
        ],
    )

onnxruntime_inspect_model_macro = macro(
    implementation = _onnxruntime_inspect_model_macro_impl,
    attrs = {
        "model_target": attr.label(
            mandatory = True,
            configurable = False,
        ),
    },
)
