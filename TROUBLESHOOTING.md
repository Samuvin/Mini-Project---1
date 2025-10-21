# Parkinson's Disease Detection System - Quick Fix Guide

## Issue: Import Errors After Setup

If you're seeing import errors like `ModuleNotFoundError: No module named 'pandas'`, you need to **activate the virtual environment** first.

## Solution

### Step 1: Activate Virtual Environment

**On macOS/Linux:**
```bash
source venv/bin/activate
```

**On Windows:**
```cmd
venv\Scripts\activate
```

You should see `(venv)` at the beginning of your terminal prompt.

### Step 2: Verify Installation

```bash
python verify.py
```

This will check if all dependencies are properly installed.

### Step 3: Download Data

```bash
python download_data.py
```

### Step 4: Train Models

```bash
python train.py
```

### Step 5: Run Web App

```bash
python webapp/app.py
```

Then open http://localhost:5000 in your browser.

## Quick Commands

```bash
# One-liner to activate and verify
source venv/bin/activate && python verify.py

# One-liner to activate and train
source venv/bin/activate && python train.py

# One-liner to activate and run webapp
source venv/bin/activate && python webapp/app.py
```

## Alternative: Use the Activation Helper

```bash
# This activates venv and shows available commands
source activate.sh
```

## Common Issues

### "venv/bin/activate: No such file or directory"
**Solution:** Run `./setup.sh` first to create the virtual environment

### "python: command not found"
**Solution:** Use `python3` instead:
```bash
python3 verify.py
python3 train.py
```

### Packages still not found after activation
**Solution:** Reinstall dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Port 5000 already in use
**Solution:** Edit `config.yaml` and change `webapp.port` to a different port (e.g., 5001)

## Full Clean Setup

If you want to start fresh:

```bash
# Remove virtual environment
rm -rf venv

# Remove any generated files
rm -rf data/processed/*.npy
rm -rf models/*.joblib

# Run setup again
./setup.sh

# Activate environment
source venv/bin/activate

# Verify installation
python verify.py

# Download data
python download_data.py

# Train models
python train.py
```

## Verify Virtual Environment is Active

When your virtual environment is active, you should see:
```bash
(venv) user@computer:~/Desktop/Spryzen/fn$
```

To check Python path:
```bash
which python
# Should output: /Users/jenishs/Desktop/Spryzen/fn/venv/bin/python
```

To check installed packages:
```bash
pip list
# Should show pandas, scikit-learn, flask, etc.
```

## Need Help?

1. Make sure virtual environment is activated
2. Run `python verify.py` to check installation
3. Check that you're in the project directory: `/Users/jenishs/Desktop/Spryzen/fn`
4. Ensure all dependencies installed: `pip list | grep pandas`

