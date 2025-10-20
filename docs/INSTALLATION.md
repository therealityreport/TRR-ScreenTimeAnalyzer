# Installation & Setup Guide

## Quick Fix (Immediate Solution)

If you're getting `ModuleNotFoundError: No module named 'screentime'`, use this quick fix:

```bash
# From the project root directory
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Now run scripts normally
python scripts/diagnose_harvest.py --video data/RHOBH-TEST.mp4 --sample 100
```

Or source the helper script:
```bash
source scripts/fix_imports.sh
```

## Proper Installation (Recommended)

Install the package in "editable" mode so Python can find the `screentime` module:

```bash
# Make sure you're in the project root
cd "/Volumes/HardDrive/SCREEN TIME ANALYZER"

# Activate your virtual environment
source .venv/bin/activate

# Install in editable mode
pip install -e .
```

After this, you can run scripts from anywhere:
```bash
python scripts/diagnose_harvest.py --video data/RHOBH-TEST.mp4 --sample 100
python scripts/validate_harvest.py data/harvest/RHOBH-TEST
```

## Verify Installation

Test that the package is installed correctly:

```bash
python -c "import screentime; print('✓ screentime module found')"
```

If you see `✓ screentime module found`, you're all set!

## Alternative: Using the Streamlit Labeler

The existing labeler app already handles imports correctly, so it works without installation:

```bash
streamlit run app/labeler.py -- \
    --harvest-dir data/harvest/RHOBH-TEST \
    --video data/RHOBH-TEST.mp4
```

## Troubleshooting

### "No module named 'screentime'" Error

**Option 1:** Set PYTHONPATH (temporary):
```bash
export PYTHONPATH="${PWD}:${PYTHONPATH}"
```

**Option 2:** Install package (permanent):
```bash
pip install -e .
```

**Option 3:** Run from project root:
```bash
# Always run scripts from the project root directory
cd "/Volumes/HardDrive/SCREEN TIME ANALYZER"
python scripts/script_name.py ...
```

### "No module named 'setuptools'" Error

Install setuptools:
```bash
pip install setuptools wheel
```

### Virtual Environment Not Working

Recreate the virtual environment:
```bash
# Deactivate if currently active
deactivate

# Remove old environment
rm -rf .venv

# Create new environment
python3.9 -m venv .venv

# Activate
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Install package
pip install -e .
```

## What Got Installed?

When you run `pip install -e .`, you're installing the `screentime` package in "editable" mode. This means:

1. Python can now find `screentime` module from anywhere
2. Changes you make to the code are immediately available (no reinstall needed)
3. All scripts can import from `screentime.*` without errors

## Scripts vs Package

- **Scripts** (`scripts/*.py`): Command-line tools you run
- **Package** (`screentime/*.py`): Core library code that scripts import

The scripts need to import from the package, so the package must be findable by Python.

## IDE Setup (VSCode, PyCharm, etc.)

If using an IDE, make sure it's configured to use the correct Python interpreter:

1. **VSCode**: Select `.venv/bin/python` as interpreter
2. **PyCharm**: Set project interpreter to `.venv/bin/python`
3. **Other IDEs**: Point to `/Volumes/HardDrive/SCREEN TIME ANALYZER/.venv/bin/python`

## Next Steps

Once installation is complete, proceed with the diagnostic workflow:

```bash
# 1. Test that everything works
chmod +x scripts/test_harvest_tools.sh
./scripts/test_harvest_tools.sh

# 2. Run diagnostics
python scripts/diagnose_harvest.py \
    --video data/RHOBH-TEST.mp4 \
    --sample 100 \
    --output diagnostics/analysis

# 3. Review results
cat diagnostics/analysis/detection_log.csv
```

---

**Still having issues?** Check that:
- You're in the project root directory
- Virtual environment is activated (you should see `(.venv)` in your prompt)
- Package is installed: `pip list | grep screentime`
