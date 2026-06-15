MODEL_FILE="$1"

# Strips ".tflite" from the end of the filename if it exists
BASE_NAME="${MODEL_FILE%.tflite}"

echo "Converting model..."
# Generates the output file using the clean base name
xxd -i "$MODEL_FILE" > "${BASE_NAME}.cc"
echo "Done!"
