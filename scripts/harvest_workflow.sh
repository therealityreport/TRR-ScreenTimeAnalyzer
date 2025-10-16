#!/bin/bash
# Helper script for harvest optimization workflow
# Usage: ./scripts/harvest_workflow.sh data/RHOBH-TEST.mp4

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

VIDEO_PATH=$1
PERSON_WEIGHTS="models/weights/yolov8n.pt"

if [ -z "$VIDEO_PATH" ]; then
    echo -e "${RED}Error: Video path required${NC}"
    echo "Usage: $0 <video_path>"
    echo "Example: $0 data/RHOBH-TEST.mp4"
    exit 1
fi

if [ ! -f "$VIDEO_PATH" ]; then
    echo -e "${RED}Error: Video file not found: $VIDEO_PATH${NC}"
    exit 1
fi

VIDEO_NAME=$(basename "$VIDEO_PATH" | sed 's/\.[^.]*$//')
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Harvest Optimization Workflow${NC}"
echo -e "${BLUE}Video: $VIDEO_NAME${NC}"
echo -e "${BLUE}========================================${NC}"

# Step 1: Diagnostics
echo -e "\n${YELLOW}Step 1: Running diagnostics...${NC}"
python scripts/diagnose_harvest.py \
    --video "$VIDEO_PATH" \
    --sample 100 \
    --output "diagnostics/${VIDEO_NAME}_diagnostic"

echo -e "${GREEN}✓ Diagnostics complete${NC}"
echo -e "Review: diagnostics/${VIDEO_NAME}_diagnostic/detection_log.csv"

# Prompt to continue
read -p "Continue with harvest? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Exiting."
    exit 0
fi

# Step 2: Baseline harvest (if original config exists)
if [ -f "configs/pipeline_original.yaml" ]; then
    echo -e "\n${YELLOW}Step 2a: Running baseline harvest with original config...${NC}"
    python scripts/harvest_faces.py \
        "$VIDEO_PATH" \
        --person-weights "$PERSON_WEIGHTS" \
        --pipeline-config configs/pipeline_original.yaml \
        --output-dir data/harvest
    
    if [ -d "data/harvest/${VIDEO_NAME}" ]; then
        mv "data/harvest/${VIDEO_NAME}" "data/harvest/${VIDEO_NAME}_original"
        echo -e "${GREEN}✓ Baseline harvest complete${NC}"
    fi
fi

# Step 3: Optimized harvest
echo -e "\n${YELLOW}Step 3: Running optimized harvest...${NC}"
python scripts/harvest_faces.py \
    "$VIDEO_PATH" \
    --person-weights "$PERSON_WEIGHTS" \
    --output-dir data/harvest

echo -e "${GREEN}✓ Optimized harvest complete${NC}"

# Step 4: Comparison (if baseline exists)
if [ -d "data/harvest/${VIDEO_NAME}_original" ]; then
    echo -e "\n${YELLOW}Step 4: Comparing results...${NC}"
    python scripts/compare_harvests.py \
        --before "data/harvest/${VIDEO_NAME}_original" \
        --after "data/harvest/${VIDEO_NAME}" \
        --output "diagnostics/${VIDEO_NAME}_comparison"
    
    echo -e "${GREEN}✓ Comparison complete${NC}"
fi

# Step 5: Validation
echo -e "\n${YELLOW}Step 5: Validating harvest...${NC}"
python scripts/validate_harvest.py \
    "data/harvest/${VIDEO_NAME}" \
    --output "diagnostics/${VIDEO_NAME}_validation"

echo -e "${GREEN}✓ Validation complete${NC}"

# Summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Workflow Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "\n${GREEN}Generated outputs:${NC}"
echo "  📊 Diagnostics: diagnostics/${VIDEO_NAME}_diagnostic/"
if [ -d "data/harvest/${VIDEO_NAME}_original" ]; then
    echo "  📈 Comparison:  diagnostics/${VIDEO_NAME}_comparison/"
fi
echo "  ✅ Validation:  diagnostics/${VIDEO_NAME}_validation/"
echo "  🎭 Harvest:     data/harvest/${VIDEO_NAME}/"

echo -e "\n${YELLOW}Next steps:${NC}"
echo "  1. Review validation report above"
echo "  2. Check sample quality: ls data/harvest/${VIDEO_NAME}/track_*/"
echo "  3. Build facebank: python scripts/build_facebank.py"
echo "  4. Run tracker: python scripts/run_tracker.py"

echo -e "\n${BLUE}========================================${NC}"
