#!/usr/bin/env bash
# DrugCLIP Autonomous Agent - 自动闭环运行脚本
set -e

INPUT_DIR="${1:-}"
OUTPUT_DIR="${2:-}"

if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: bash run.sh <input_dir> <output_dir> [max_iterations=5]"
    exit 1
fi

MAX_ITER="${3:-5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "DrugCLIP Autonomous Agent"
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "Iter:   $MAX_ITER"
echo "============================================================"

# Auto-detect Python with rdkit installed
detect_python() {
    # Try common conda env paths first
    for env in \
        "$HOME/python_pkgs/conda_envs/ml/bin/python3" \
        "$HOME/miniconda3/envs/ml/bin/python3" \
        "$HOME/anaconda3/envs/ml/bin/python3" \
        "$HOME/conda/envs/ml/bin/python3"
    do
        if [ -x "$env" ] && "$env" -c "import rdkit; print(rdkit.__version__)" >/dev/null 2>&1; then
            echo "$env"
            return 0
        fi
    done

    # Try system python3 if it has rdkit
    if python3 -c "import rdkit" 2>/dev/null; then
        echo "python3"
        return 0
    fi

    # Try conda base
    if [ -x "$HOME/conda/bin/python" ]; then
        if "$HOME/conda/bin/python" -c "import rdkit" 2>/dev/null; then
            echo "$HOME/conda/bin/python"
            return 0
        fi
    fi

    return 1
}

PYTHON=$(detect_python) || {
    echo ""
    echo "ERROR: No Python with rdkit>=2022.03 found."
    echo ""
    echo "Please install rdkit:"
    echo "  conda install -c conda-forge rdkit=2022.03"
    echo "  # or"
    echo "  pip install rdkit"
    echo ""
    echo "Or set up a conda environment:"
    echo "  conda create -n ml python=3.10 rdkit scikit-learn pandas numpy scipy"
    echo "  conda activate ml"
    exit 1
}

PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH" $PYTHON src/main.py \
    "$INPUT_DIR" \
    "$OUTPUT_DIR" \
    "$MAX_ITER"

echo ""
echo "============================================================"
echo "Done. Result: $OUTPUT_DIR/result.zip"
echo "============================================================"