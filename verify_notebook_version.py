#!/usr/bin/env python3
"""
Verify which version of model_comparison.ipynb is loaded.
Run this on the remote server to check if changes synced correctly.
"""
import json
from pathlib import Path


def verify_notebook(notebook_path):
    """Check if notebook has the 3-panel confusion matrix changes."""
    print(f"🔍 Checking: {notebook_path}")
    print(f"📂 Absolute path: {Path(notebook_path).absolute()}")
    print(f"📄 File exists: {Path(notebook_path).exists()}")

    nb_path = Path(notebook_path)
    if not nb_path.exists():
        print("❌ File not found!")
        return False

    with nb_path.open('r') as f:
        nb = json.load(f)

    print(f"\n📊 Notebook has {len(nb['cells'])} cells")

    # Check Cell 16 (Data Preparation)
    print("\n" + "="*60)
    print("CELL 16: Data Preparation")
    print("="*60)
    cell_16 = nb['cells'][16]
    cell_16_source = ''.join(cell_16['source'])

    checks_16 = {
        'internvl_nq_batch_df': 'internvl_nq_batch_df' in cell_16_source,
        'image_stem normalization': 'image_stem' in cell_16_source,
        'Column name normalization': "image_name" in cell_16_source,
        'Line count': len(cell_16['source'])
    }

    for check, result in checks_16.items():
        status = "✅" if result else "❌"
        print(f"{status} {check}: {result}")

    # Check Cell 18 (Confusion Matrix)
    print("\n" + "="*60)
    print("CELL 18: Confusion Matrix Visualization")
    print("="*60)
    cell_18 = nb['cells'][18]
    cell_18_source = ''.join(cell_18['source'])

    checks_18 = {
        'abbreviate_doctype function': 'def abbreviate_doctype' in cell_18_source,
        '3-panel subplot': 'subplots(1, 3' in cell_18_source,
        'ax3 (3rd panel)': ', ax3)' in cell_18_source,
        'InternVL3-NonQuantized CM': 'internvl_nq_cm' in cell_18_source,
        'pd.crosstab (non-square)': 'pd.crosstab' in cell_18_source,
        'CTP_INSUR abbreviation': 'CTP_INSUR' in cell_18_source,
        'Line count': len(cell_18['source'])
    }

    for check, result in checks_18.items():
        status = "✅" if result else "❌"
        print(f"{status} {check}: {result}")

    # Overall verdict
    print("\n" + "="*60)
    all_passed = (
        checks_16['internvl_nq_batch_df'] and
        checks_16['Line count'] >= 60 and
        checks_18['abbreviate_doctype function'] and
        checks_18['3-panel subplot'] and
        checks_18['ax3 (3rd panel)'] and
        checks_18['InternVL3-NonQuantized CM'] and
        checks_18['Line count'] >= 180
    )

    if all_passed:
        print("✅ PASS: Notebook has all 3-panel confusion matrix changes!")
        print("Expected output:")
        print("  - 3 confusion matrices side-by-side")
        print("  - Abbreviated labels (CTP_INSUR, E-TICKET, etc.)")
        print("  - InternVL3-NonQuantized as 3rd panel")
    else:
        print("❌ FAIL: Notebook is missing changes!")
        print("Expected:")
        print(f"  - Cell 16: ≥60 lines (found {checks_16['Line count']})")
        print(f"  - Cell 18: ≥180 lines (found {checks_18['Line count']})")
        print("Action needed: Pull latest changes from git")

    print("="*60)
    return all_passed

if __name__ == "__main__":
    # Check both possible locations
    paths = [
        "/home/jovyan/_LMM_POC/model_comparison.ipynb",  # Remote server
        "/Users/tod/Desktop/LMM_POC/model_comparison.ipynb"  # Local Mac
    ]

    for path in paths:
        if Path(path).exists():
            verify_notebook(path)
            print("\n")
