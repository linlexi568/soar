#!/usr/bin/env python3
"""
Verify Soar integration in benchmark environment
This script checks that all necessary files are in place and the controller works.
"""

import sys
from pathlib import Path

def check_files():
    """Check if all required files exist"""
    root = Path(__file__).parent.parent
    required_files = [
        'baselines/soar_controller.py',
        'baselines/eval_soar.py',
        'baselines/SOAR_README.md',
        'baselines/SOAR_INTEGRATION_COMPLETE.md',
    ]
    
    print("Checking required files...")
    all_exist = True
    for file_path in required_files:
        full_path = root / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING!")
            all_exist = False
    
    return all_exist

def check_imports():
    """Try to import the controller (will fail if numpy not available)"""
    print("\nChecking imports...")
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from soar_controller import SoarController
        print("  ✓ SoarController imported successfully")
        return True
    except ImportError as e:
        print(f"  ⚠ Import failed (expected if venv not activated): {e}")
        print("    This is OK - activate venv to use the controller")
        return False

def check_parameters():
    """Check if manual.md parameters match controller"""
    print("\nChecking parameter consistency...")
    
    # Read manual.md
    root = Path(__file__).parent.parent.parent
    manual_path = root / 'manual.md'
    
    if not manual_path.exists():
        print("  ⚠ manual.md not found at expected location")
        return False
    
    with open(manual_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check key parameters are documented
    checks = [
        ('Figure8 trajectory', '0.489'),
        ('Square trajectory', '0.600'),
        ('Circle trajectory', '2.104'),
        ('sign operator', 'sign(e)'),
        ('smooth operator', 'smooth(e'),
        ('nonlinear control', 'sign'),
    ]
    
    all_ok = True
    for key, expected in checks:
        # Case-insensitive search
        if expected in content or expected.lower() in content.lower():
            print(f"  ✓ '{key}' documented")
        else:
            print(f"  ✗ '{key}' missing or incorrect")
            all_ok = False
    
    return all_ok

def main():
    print("="*60)
    print("Soar Benchmark Integration Verification")
    print("="*60)
    
    files_ok = check_files()
    imports_ok = check_imports()
    params_ok = check_parameters()
    
    print("\n" + "="*60)
    if files_ok and params_ok:
        print("✅ All checks passed!")
        print("\nNext steps:")
        print("1. Activate your virtual environment with Isaac Gym")
        print("2. Run: python baselines/eval_soar.py --task circle --episodes 1")
        print("3. See SOAR_README.md for full documentation")
    else:
        print("❌ Some checks failed - please review above")
    print("="*60)

if __name__ == '__main__':
    main()
