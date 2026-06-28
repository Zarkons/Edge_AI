load("@rules_python//python:defs.bzl", "py_binary")

def _inspect_model_macro_impl(name, model_target, visibility = None):
    py_binary(
        name = name,
        srcs = ["//modules/helpers:inspect_model.py"],
        main = "//modules/helpers:inspect_model.py",
        args = [
            "--model_path=$(rootpath {})".format(model_target),
        ],
        data = [model_target],
        visibility = visibility,
        deps = [
            "@rules_python//python/runfiles",
        ],
    )

inspect_model_macro = macro(
    implementation = _inspect_model_macro_impl,
    attrs = {
        "model_target": attr.label(
            mandatory = True,
            configurable = False,
        ),
    },
)
