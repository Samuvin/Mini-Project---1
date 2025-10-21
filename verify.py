#!/usr/bin/env python3
"""
Verification script to check if all components are properly installed.
"""

import sys
import os
from pathlib import Path

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"  {GREEN}✓{RESET} {description}")
        return True
    else:
        print(f"  {RED}✗{RESET} {description} - MISSING")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists."""
    if Path(dirpath).is_dir():
        print(f"  {GREEN}✓{RESET} {description}")
        return True
    else:
        print(f"  {RED}✗{RESET} {description} - MISSING")
        return False

def check_import(module_name, description):
    """Check if a Python module can be imported."""
    try:
        __import__(module_name)
        print(f"  {GREEN}✓{RESET} {description}")
        return True
    except ImportError:
        print(f"  {RED}✗{RESET} {description} - NOT INSTALLED")
        return False

def main():
    """Main verification function."""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Parkinson's Disease Detection System - Verification{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    all_checks_passed = True
    
    # Check project structure
    print(f"{YELLOW}Checking Project Structure...{RESET}")
    checks = [
        ('README.md', 'Main README'),
        ('QUICKSTART.md', 'Quick Start Guide'),
        ('IMPLEMENTATION_SUMMARY.md', 'Implementation Summary'),
        ('config.yaml', 'Configuration file'),
        ('requirements.txt', 'Requirements file'),
        ('train.py', 'Training script'),
        ('.gitignore', 'Git ignore file'),
    ]
    
    for filepath, desc in checks:
        if not check_file_exists(filepath, desc):
            all_checks_passed = False
    
    # Check directories
    print(f"\n{YELLOW}Checking Directory Structure...{RESET}")
    dirs = [
        ('src', 'Source code directory'),
        ('src/data', 'Data modules'),
        ('src/features', 'Feature extraction modules'),
        ('src/models', 'Model modules'),
        ('src/evaluation', 'Evaluation modules'),
        ('src/utils', 'Utility modules'),
        ('webapp', 'Web application'),
        ('webapp/templates', 'HTML templates'),
        ('webapp/static', 'Static files'),
        ('webapp/api', 'API modules'),
        ('tests', 'Test suite'),
        ('data', 'Data directory'),
        ('models', 'Models directory'),
        ('notebooks', 'Jupyter notebooks'),
    ]
    
    for dirpath, desc in dirs:
        if not check_directory_exists(dirpath, desc):
            all_checks_passed = False
    
    # Check source modules
    print(f"\n{YELLOW}Checking Source Modules...{RESET}")
    source_files = [
        ('src/__init__.py', 'Source package init'),
        ('src/data/data_loader.py', 'Data loader'),
        ('src/data/preprocessor.py', 'Data preprocessor'),
        ('src/features/speech_features.py', 'Speech features'),
        ('src/features/handwriting_features.py', 'Handwriting features'),
        ('src/features/gait_features.py', 'Gait features'),
        ('src/models/logistic_regression.py', 'Logistic Regression model'),
        ('src/models/svm_model.py', 'SVM model'),
        ('src/evaluation/metrics.py', 'Evaluation metrics'),
        ('src/utils/config.py', 'Configuration utilities'),
    ]
    
    for filepath, desc in source_files:
        if not check_file_exists(filepath, desc):
            all_checks_passed = False
    
    # Check web application files
    print(f"\n{YELLOW}Checking Web Application...{RESET}")
    webapp_files = [
        ('webapp/app.py', 'Flask application'),
        ('webapp/api/predict.py', 'Prediction API'),
        ('webapp/templates/index.html', 'Home page template'),
        ('webapp/templates/predict.html', 'Prediction page template'),
        ('webapp/templates/about.html', 'About page template'),
        ('webapp/templates/documentation.html', 'Documentation page template'),
        ('webapp/static/css/style.css', 'CSS stylesheet'),
        ('webapp/static/js/main.js', 'Main JavaScript'),
        ('webapp/static/js/predict.js', 'Prediction JavaScript'),
    ]
    
    for filepath, desc in webapp_files:
        if not check_file_exists(filepath, desc):
            all_checks_passed = False
    
    # Check test files
    print(f"\n{YELLOW}Checking Test Suite...{RESET}")
    test_files = [
        ('tests/__init__.py', 'Tests package init'),
        ('tests/test_data.py', 'Data tests'),
        ('tests/test_models.py', 'Model tests'),
    ]
    
    for filepath, desc in test_files:
        if not check_file_exists(filepath, desc):
            all_checks_passed = False
    
    # Check Python dependencies
    print(f"\n{YELLOW}Checking Python Dependencies...{RESET}")
    dependencies = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'scikit-learn'),
        ('flask', 'Flask'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('yaml', 'PyYAML'),
        ('joblib', 'Joblib'),
        ('scipy', 'SciPy'),
        ('imblearn', 'imbalanced-learn'),
    ]
    
    for module, desc in dependencies:
        if not check_import(module, desc):
            all_checks_passed = False
    
    # Optional dependencies
    print(f"\n{YELLOW}Checking Optional Dependencies...{RESET}")
    optional_deps = [
        ('pytest', 'pytest'),
        ('jupyter', 'Jupyter'),
        ('librosa', 'librosa'),
    ]
    
    for module, desc in optional_deps:
        check_import(module, f"{desc} (optional)")
    
    # Final summary
    print(f"\n{BLUE}{'='*70}{RESET}")
    if all_checks_passed:
        print(f"{GREEN}✓ All critical components verified successfully!{RESET}")
        print(f"\n{GREEN}System is ready to use.{RESET}")
        print(f"\n{YELLOW}Next steps:{RESET}")
        print(f"  1. Run training: {BLUE}python train.py{RESET}")
        print(f"  2. Start web app: {BLUE}python webapp/app.py{RESET}")
        print(f"  3. Or run tests: {BLUE}pytest tests/ -v{RESET}")
    else:
        print(f"{RED}✗ Some components are missing.{RESET}")
        print(f"\n{YELLOW}Please run setup:{RESET}")
        print(f"  Unix/Mac: {BLUE}./setup.sh{RESET}")
        print(f"  Windows: {BLUE}setup.bat{RESET}")
    
    print(f"{BLUE}{'='*70}{RESET}\n")
    
    return 0 if all_checks_passed else 1

if __name__ == '__main__':
    sys.exit(main())

