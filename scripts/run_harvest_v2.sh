#!/bin/bash
# Lightweight harvest runner for RHOBH-TEST-v2
# Usage: (after activating the venv)
#   ./scripts/run_harvest_v2.sh
#
# The script keeps CPU usage manageable by forcing single-threaded
# math libraries and using the fast harvest path (stride=2, 640px detector).

set -euo pipefail

VIDEO_PATH="data/RHOBH-TEST-v2.mp4"
OUTPUT_ROOT="data/harvest"
HARVEST_NAME="RHOBH-TEST-v2"
PERSON_WEIGHTS="models/weights/yolov8n.pt"

if ! command -v python >/dev/null 2>&1; then
    echo "python is not on PATH. Activate the virtualenv first:"
    echo "  source \"/Volumes/HardDrive/SCREEN TIME ANALYZER/.venv/bin/activate\""
    exit 1
fi

if [ ! -f "$VIDEO_PATH" ]; then
    echo "Video file not found: $VIDEO_PATH"
    exit 1
fi

echo "Limiting CPU-intensive libraries to 1 thread…"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export ONNX_THREADPOOL_SPIN_CONTROL=0

OUTPUT_DIR="${OUTPUT_ROOT}/${HARVEST_NAME}"
if [ -d "$OUTPUT_DIR" ]; then
    echo "Removing previous harvest output at ${OUTPUT_DIR}"
    rm -rf "$OUTPUT_DIR"
fi

echo "Running harvest for ${VIDEO_PATH} → ${OUTPUT_DIR}"
python scripts/harvest_faces.py \
    "$VIDEO_PATH" \
    --person-weights "$PERSON_WEIGHTS" \
    --output-dir "$OUTPUT_ROOT" \
    --fast \
    --person-conf 0.10 \
    --retina-det-size 640 640 \
    --onnx-providers CPUExecutionProvider

echo "Harvest complete. New tracks written to ${OUTPUT_DIR}"
