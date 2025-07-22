#!/usr/bin/env python3
"""
Model Organization Script

This script moves model files from the root directory to the models/
directory and renames them according to the new naming convention.

Author: MyGymPal.ai Team
Date: 2024
"""

import os
import shutil
from pathlib import Path

def organize_model_files():
    """
    Organize model files by moving them to the models directory
    and renaming them according to the new convention.
    """
    print("=" * 60)
    print("MyGymPal.ai - Model Organization Script")
    print("=" * 60)
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print("✓ Models directory created/verified")
    
    # Define file mappings (old_name -> new_name)
    file_mappings = {
        'svm_81.11.h5': 'custom_ann_model.h5',
        'label_encoder_bicepcurlphase.pkl': 'label_encoder_bicepcurlphase.pkl',
        'label_encoder_orientation.pkl': 'label_encoder_orientation.pkl',
        'encoder.pkl': 'encoder.pkl',
        'movenet_singlepose_lightning.tflite': 'movenet_singlepose_lightning.tflite'
    }
    
    moved_files = []
    missing_files = []
    
    for old_name, new_name in file_mappings.items():
        source_path = Path(old_name)
        dest_path = models_dir / new_name
        
        if source_path.exists():
            try:
                # Move file to models directory
                shutil.move(str(source_path), str(dest_path))
                print(f"✓ Moved: {old_name} -> models/{new_name}")
                moved_files.append(new_name)
            except Exception as e:
                print(f"❌ Error moving {old_name}: {e}")
        else:
            print(f"⚠️  File not found: {old_name}")
            missing_files.append(old_name)
    
    # Create a summary
    print("\n" + "=" * 60)
    print("ORGANIZATION SUMMARY")
    print("=" * 60)
    print(f"✓ Successfully moved {len(moved_files)} files to models/ directory")
    
    if missing_files:
        print(f"⚠️  {len(missing_files)} files were not found:")
        for file in missing_files:
            print(f"   - {file}")
    
    # Update configuration to reflect new paths
    print("\nUpdating configuration files...")
    update_configuration_files()
    
    print("\n" + "=" * 60)
    print("✅ Model organization completed!")
    print("=" * 60)
    print("\nModel files are now organized in the models/ directory.")
    print("The application will automatically look for models in this location.")

def update_configuration_files():
    """
    Update configuration files to reflect new model paths.
    """
    # Update config/settings.py
    config_file = Path("config/settings.py")
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                content = f.read()
            
            # Update model paths to include models/ prefix
            content = content.replace(
                "'custom_ann': 'custom_ann_model.h5'",
                "'custom_ann': 'models/custom_ann_model.h5'"
            )
            content = content.replace(
                "'label_encoder_bicepcurl': 'label_encoder_bicepcurlphase.pkl'",
                "'label_encoder_bicepcurl': 'models/label_encoder_bicepcurlphase.pkl'"
            )
            content = content.replace(
                "'label_encoder_orientation': 'label_encoder_orientation.pkl'",
                "'label_encoder_orientation': 'models/label_encoder_orientation.pkl'"
            )
            content = content.replace(
                "'encoder': 'encoder.pkl'",
                "'encoder': 'models/encoder.pkl'"
            )
            content = content.replace(
                "'movenet': 'movenet_singlepose_lightning.tflite'",
                "'movenet': 'models/movenet_singlepose_lightning.tflite'"
            )
            
            with open(config_file, 'w') as f:
                f.write(content)
            
            print("✓ Updated config/settings.py")
        except Exception as e:
            print(f"❌ Error updating config/settings.py: {e}")
    
    # Update run.py
    run_file = Path("run.py")
    if run_file.exists():
        try:
            with open(run_file, 'r') as f:
                content = f.read()
            
            # Update default model paths
            content = content.replace(
                "default='custom_ann_model.h5'",
                "default='models/custom_ann_model.h5'"
            )
            content = content.replace(
                "'label_encoder_bicepcurl': 'label_encoder_bicepcurlphase.pkl'",
                "'label_encoder_bicepcurl': 'models/label_encoder_bicepcurlphase.pkl'"
            )
            content = content.replace(
                "'label_encoder_orientation': 'label_encoder_orientation.pkl'",
                "'label_encoder_orientation': 'models/label_encoder_orientation.pkl'"
            )
            content = content.replace(
                "print('- custom_ann_model.h5')",
                "print('- models/custom_ann_model.h5')"
            )
            content = content.replace(
                "print('- label_encoder_bicepcurlphase.pkl')",
                "print('- models/label_encoder_bicepcurlphase.pkl')"
            )
            content = content.replace(
                "print('- label_encoder_orientation.pkl')",
                "print('- models/label_encoder_orientation.pkl')"
            )
            
            with open(run_file, 'w') as f:
                f.write(content)
            
            print("✓ Updated run.py")
        except Exception as e:
            print(f"❌ Error updating run.py: {e}")
    
    # Update setup.py
    setup_file = Path("setup.py")
    if setup_file.exists():
        try:
            with open(setup_file, 'r') as f:
                content = f.read()
            
            # Update model file paths
            content = content.replace(
                "'custom_ann_model.h5'",
                "'models/custom_ann_model.h5'"
            )
            content = content.replace(
                "'label_encoder_bicepcurlphase.pkl'",
                "'models/label_encoder_bicepcurlphase.pkl'"
            )
            content = content.replace(
                "'label_encoder_orientation.pkl'",
                "'models/label_encoder_orientation.pkl'"
            )
            
            with open(setup_file, 'w') as f:
                f.write(content)
            
            print("✓ Updated setup.py")
        except Exception as e:
            print(f"❌ Error updating setup.py: {e}")

def clean_root_directory():
    """
    Clean up the root directory by removing old files and organizing data.
    """
    print("\nCleaning root directory...")
    
    # Files to remove from root (after moving to appropriate directories)
    files_to_remove = [
        'SVM.py',  # Old SVM module
        'MovenetModule.py',  # Old module
        'AITrainerProject.py',  # Old module
        'main.py',  # Old main file
        'test.py',  # Test file
        'fyp.ipynb',  # Jupyter notebook
        'qodana.yaml',  # IDE config
        'pyvenv.cfg',  # Virtual environment
        'variables.index',  # TensorFlow variables
        'variables.data-00000-of-00001',  # TensorFlow variables
        'variables (1).index',  # TensorFlow variables
        'variables (1).data-00000-of-00001',  # TensorFlow variables
        'saved_model.pb',  # TensorFlow model
        'saved_model (1).pb',  # TensorFlow model
        'keras_metadata.pb',  # Keras metadata
        'keras_metadata (1).pb',  # Keras metadata
        'fingerprint.pb',  # TensorFlow fingerprint
        'fingerprint (1).pb',  # TensorFlow fingerprint
        'nn.h5',  # Neural network model
        'model.weights.h5',  # Model weights
        'output.png',  # Output image
        'output.gif',  # Output gif
        'nolt.gif',  # Gif file
        'frames.zip',  # Compressed frames
        'keypoints_data.csv',  # Data file
        'keypoints_data1.csv',  # Data file
        'keypoints_data2.csv',  # Data file
        'keypoints_data3(all vids).csv',  # Data file
        'keypointsfinal.csv',  # Data file
        'keypqqoints_data.csv',  # Data file
        'correct_processed.csv',  # Data file
        'incorrect_processed.csv'  # Data file
    ]
    
    # Create data directory for CSV files
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    removed_count = 0
    moved_count = 0
    
    for file_name in files_to_remove:
        file_path = Path(file_name)
        if file_path.exists():
            try:
                if file_name.endswith('.csv'):
                    # Move CSV files to data directory
                    shutil.move(str(file_path), str(data_dir / file_name))
                    print(f"✓ Moved to data/: {file_name}")
                    moved_count += 1
                else:
                    # Remove other files
                    file_path.unlink()
                    print(f"✓ Removed: {file_name}")
                    removed_count += 1
            except Exception as e:
                print(f"❌ Error processing {file_name}: {e}")
    
    print(f"\n✓ Cleaned root directory: {removed_count} files removed, {moved_count} files moved to data/")

if __name__ == "__main__":
    try:
        # Organize model files
        organize_model_files()
        
        # Clean root directory
        clean_root_directory()
        
        print("\n🎉 Root directory cleanup completed!")
        print("The project is now properly organized with:")
        print("- Model files in models/ directory")
        print("- Data files in data/ directory")
        print("- Clean root directory")
        
    except Exception as e:
        print(f"❌ Error during organization: {e}")
        import traceback
        traceback.print_exc() 