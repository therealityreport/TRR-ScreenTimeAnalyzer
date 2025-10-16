#!/bin/bash
# Quick test of the new harvest optimization tools
# Run this to verify everything is working

echo "========================================="
echo "Testing Harvest Optimization Tools"
echo "========================================="

# Check if video exists
if [ ! -f "data/RHOBH-TEST.mp4" ]; then
    echo "❌ Test video not found: data/RHOBH-TEST.mp4"
    exit 1
fi

echo "✓ Test video found"

# Test 1: Diagnostic script
echo ""
echo "Test 1: Running diagnostics on 10 frames..."
python scripts/diagnose_harvest.py \
    --video data/RHOBH-TEST.mp4 \
    --sample 10 \
    --output diagnostics/test_run

if [ $? -eq 0 ]; then
    echo "✓ Diagnostics completed successfully"
else
    echo "❌ Diagnostics failed"
    exit 1
fi

# Test 2: Config validation
echo ""
echo "Test 2: Checking configuration files..."
for config in configs/pipeline.yaml configs/pipeline_original.yaml configs/pipeline_permissive.yaml; do
    if [ -f "$config" ]; then
        echo "  ✓ Found: $config"
    else
        echo "  ❌ Missing: $config"
        exit 1
    fi
done

# Test 3: Check if harvest has already been run
echo ""
echo "Test 3: Checking for existing harvest..."
if [ -d "data/harvest/RHOBH-TEST" ]; then
    echo "  ✓ Harvest directory exists"
    
    # Test validation
    echo ""
    echo "Test 4: Running validation on existing harvest..."
    python scripts/validate_harvest.py \
        data/harvest/RHOBH-TEST \
        --output diagnostics/test_validation
    
    if [ $? -eq 0 ]; then
        echo "✓ Validation completed successfully"
    else
        echo "❌ Validation failed"
        exit 1
    fi
else
    echo "  ℹ️  No existing harvest found (run harvest_faces.py first)"
fi

echo ""
echo "========================================="
echo "✓ All tests passed!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Run full diagnostic: python scripts/diagnose_harvest.py --video data/RHOBH-TEST.mp4 --sample 100"
echo "  2. Run harvest: python scripts/harvest_faces.py data/RHOBH-TEST.mp4 --person-weights models/weights/yolov8n.pt"
echo "  3. Validate results: python scripts/validate_harvest.py data/harvest/RHOBH-TEST"
echo ""
echo "Or use the automated workflow:"
echo "  ./scripts/harvest_workflow.sh data/RHOBH-TEST.mp4"
