"""
Finn Feature Engineering Script
==============================
Feature engineering for Titanic dataset with:
- Missing value handling (Age by Pclass+Sex)
- One-Hot encoding of categoricals
- StandardScaler for numericals
- Train/Test split (80/20)

Usage:
    python finn_script.py --input <path> --output-dir <dir>
"""