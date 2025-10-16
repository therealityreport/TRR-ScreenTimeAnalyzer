#!/bin/bash
# Quick fix for Python import errors
# Run this from the project root: source scripts/fix_imports.sh

export PYTHONPATH="${PWD}:${PYTHONPATH}"
echo "✓ PYTHONPATH updated: ${PYTHONPATH}"
echo ""
echo "You can now run:"
echo "  python scripts/diagnose_harvest.py --video data/RHOBH-TEST.mp4 --sample 100"
echo "  python scripts/validate_harvest.py data/harvest/RHOBH-TEST"
echo "  python scripts/compare_harvests.py --before ... --after ..."
