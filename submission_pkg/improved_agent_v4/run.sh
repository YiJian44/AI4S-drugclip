#!/bin/bash
# DrugCLIP Agent Submission Runner
# Usage: bash run.sh <input_dir> <output_dir>

set -e

INPUT_DIR="${1:-./input}"
OUTPUT_DIR="${2:-./output}"

echo "[$(date)] DrugCLIP Agent Starting"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"
cd "$(dirname $0)"

export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Find Python with rdkit
if [ -f "/home/yijian/miniforge3/envs/ml/bin/python" ]; then
    PYTHON="/home/yijian/miniforge3/envs/ml/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

echo "Using Python: $PYTHON"

$PYTHON "$(pwd)/src/main.py" "$INPUT_DIR" "$OUTPUT_DIR"

echo "[$(date)] DrugCLIP Agent Finished"