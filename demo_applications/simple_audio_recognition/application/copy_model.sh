#!/bin/bash
set -e

# Capture the dynamic sandbox path passed by Bazel's args configuration
SOURCE_FILE_PATH="$1"

# The permanent destination path in your real source tree
TARGET_DIR="${BUILD_WORKSPACE_DIRECTORY}/demo_applications/simple_audio_recognition/application/tflite_model"

# Verify that Bazel actually passed the file path argument
if [ -z "$SOURCE_FILE_PATH" ]; then
    echo "Error: No input path provided by Bazel target."
    exit 1
fi

echo "Copying file [${SOURCE_FILE_PATH}] to workspace..."

# Execute the copy operation directly into your source folder
cp -f "$SOURCE_FILE_PATH" "${TARGET_DIR}/audio_recognition_model.cc"

echo "Success! Model written to: ${TARGET_DIR}/audio_recognition_model.cc"
