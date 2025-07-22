#!/usr/bin/env python3
"""
MyGymPal.ai - Setup Script

This script helps users set up the MyGymPal.ai fitness application
by checking dependencies, creating necessary directories, and validating
model files.

Author: MyGymPal.ai Team
Date: 2024
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_banner():
    """Print the application banner."""
    print("=" * 60)
    print("MyGymPal.ai - AI-Powered Fitness Application")
    print("Setup and Installation Script")
    print("=" * 60)
    print()

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    else:
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        return True

def check_dependencies():
    """Check if required dependencies are installed."""
    print("\nChecking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'opencv-python', 'tensorflow', 
        'scikit-learn', 'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install them using: pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """Create necessary directories if they don't exist."""
    print("\nCreating directories...")
    
    directories = [
        'models',
        'data',
        'output',
        'logs',
        'tests'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_model_files():
    """Check if required model files exist."""
    print("\nChecking model files...")
    
    required_files = [
        'movenet_singlepose_lightning.tflite',
        'custom_ann_model.h5',
        'label_encoder_bicepcurlphase.pkl',
        'label_encoder_orientation.pkl'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - Missing")
            missing_files.append(file)
    
    if missing_files:
        print(f"\nMissing model files: {', '.join(missing_files)}")
        print("Please download the required model files and place them in the project root.")
        return False
    
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\nInstalling dependencies...")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def run_tests():
    """Run basic tests to verify installation."""
    print("\nRunning basic tests...")
    
    try:
        # Test imports
        sys.path.append('src')
        from core.pose_detection import PoseDetector
        from core.visualization import Visualizer
        from ml.custom_classifier import CustomClassifier
        print("✅ All modules can be imported")
        
        # Test configuration
        from config.settings import MODEL_PATHS, COLORS
        print("✅ Configuration loaded successfully")
        
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup function."""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check dependencies
    if not check_dependencies():
        print("\nInstalling missing dependencies...")
        if not install_dependencies():
            print("❌ Failed to install dependencies. Please install them manually.")
            sys.exit(1)
    
    # Check model files
    if not check_model_files():
        print("\n⚠️  Some model files are missing.")
        print("The application may not work properly without all model files.")
        print("Please download the required model files and place them in the project root.")
    
    # Run tests
    if not run_tests():
        print("❌ Basic tests failed. Please check the installation.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ Setup completed successfully!")
    print("=" * 60)
    print("\nTo run the application:")
    print("  python run.py")
    print("\nFor help:")
    print("  python run.py --help")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main() 