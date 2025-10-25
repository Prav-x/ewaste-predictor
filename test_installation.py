#!/usr/bin/env python3
"""
Installation Test Script for E-Waste AI Predictor

This script tests if all required dependencies are installed correctly
and the application can run without errors.

Usage:
    python test_installation.py
"""

import sys
import importlib
import os

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        if package_name:
            importlib.import_module(module_name, package_name)
        else:
            importlib.import_module(module_name)
        return True, None
    except ImportError as e:
        return False, str(e)

def test_file_exists(file_path):
    """Test if a required file exists"""
    return os.path.exists(file_path), f"File not found: {file_path}"

def main():
    """Main test function"""
    print("Testing E-Waste AI Predictor Installation")
    print("="*50)
    
    # Test results
    tests_passed = 0
    total_tests = 0
    
    # Test Python version
    total_tests += 1
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print("OK Python version:", f"{python_version.major}.{python_version.minor}.{python_version.micro}")
        tests_passed += 1
    else:
        print("ERROR Python version:", f"{python_version.major}.{python_version.minor}.{python_version.micro}")
        print("   Required: Python 3.8 or higher")
    
    # Test required modules
    required_modules = [
        ('tensorflow', 'TensorFlow'),
        ('streamlit', 'Streamlit'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('plotly', 'Plotly'),
        ('pandas', 'Pandas')
    ]
    
    print("\nTesting Required Modules:")
    for module, name in required_modules:
        total_tests += 1
        success, error = test_import(module)
        if success:
            print(f"OK {name}")
            tests_passed += 1
        else:
            print(f"ERROR {name}: {error}")
    
    # Test required files
    required_files = [
        'app.py',
        'ewaste_predictor.py',
        'requirements.txt',
        'train_model.py'
    ]
    
    print("\nTesting Required Files:")
    for file_path in required_files:
        total_tests += 1
        success, error = test_file_exists(file_path)
        if success:
            print(f"OK {file_path}")
            tests_passed += 1
        else:
            print(f"ERROR {file_path}: {error}")
    
    # Test dataset structure
    total_tests += 1
    dataset_dir = "ewaste_dataset"
    if os.path.exists(dataset_dir):
        train_dir = os.path.join(dataset_dir, "train")
        val_dir = os.path.join(dataset_dir, "val")
        test_dir = os.path.join(dataset_dir, "test")
        
        if all(os.path.exists(d) for d in [train_dir, val_dir, test_dir]):
            print(f"OK Dataset structure: {dataset_dir}/")
            tests_passed += 1
        else:
            print(f"ERROR Dataset structure: Missing train/val/test directories")
    else:
        print(f"ERROR Dataset directory not found: {dataset_dir}")
    
    # Test model creation
    total_tests += 1
    try:
        from ewaste_predictor import EWastePredictor
        predictor = EWastePredictor()
        predictor.create_model()
        print("OK Model creation test passed")
        tests_passed += 1
    except Exception as e:
        print(f"ERROR Model creation test failed: {str(e)}")
    
    # Print summary
    print("\n" + "="*50)
    print(f"Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("SUCCESS: All tests passed! Installation is successful.")
        print("\nYou can now run the application:")
        print("   streamlit run app.py")
        print("\nOr train the model:")
        print("   python train_model.py")
    else:
        print("ERROR: Some tests failed. Please check the errors above.")
        print("\nTo fix installation issues:")
        print("   1. Install missing dependencies: pip install -r requirements.txt")
        print("   2. Check Python version (3.8+ required)")
        print("   3. Ensure all files are present")
        print("   4. Verify dataset structure")
    
    print("="*50)
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
