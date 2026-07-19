#!/bin/bash
MODEL_FILE="$1"
echo "Making a graph for model..."
open -a Netron "$MODEL_FILE"
echo "Done!"