#!/usr/bin/env python3
"""
File Organization Script

This script organizes all .pb, .h5, and .csv files into appropriate directories
for a cleaner project structure.

Author: MyGymPal.ai Team
Date: 2024
"""

import os
import shutil
from pathlib import Path
import glob

def organize_files():
    """
    Organize all files by type into appropriate directories.
    """
    print("=" * 60)
    print("MyGymPal.ai - File Organization Script")
    print("=" * 60)
    
    # Create directories
    directories = {
        'models': Path("models"),
        'data': Path("data"),
        'tensorflow_models': Path("models/tensorflow"),
        'keras_models': Path("models/keras"),
        'label_encoders': Path("models/label_encoders"),
        'raw_data': Path("data/raw"),
        'processed_data': Path("data/processed"),
        'temp': Path("temp")
    }
    
    # Create all directories
    for dir_name, dir_path in directories.items():
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")
    
    # File organization mappings
    file_mappings = {
        # TensorFlow model files (.pb)
        'tensorflow_models': {
            'saved_model.pb': 'saved_model.pb',
            'saved_model (1).pb': 'saved_model_backup.pb',
            'keras_metadata.pb': 'keras_metadata.pb',
            'keras_metadata (1).pb': 'keras_metadata_backup.pb',
            'fingerprint.pb': 'fingerprint.pb',
            'fingerprint (1).pb': 'fingerprint_backup.pb',
            'variables.index': 'variables.index',
            'variables.data-00000-of-00001': 'variables.data-00000-of-00001',
            'variables (1).index': 'variables_backup.index',
            'variables (1).data-00000-of-00001': 'variables_backup.data-00000-of-00001'
        },
        
        # Keras model files (.h5)
        'keras_models': {
            'svm_81.11.h5': 'custom_ann_model.h5',
            'nn.h5': 'neural_network_model.h5',
            'model.weights.h5': 'model_weights.h5'
        },
        
        # Label encoder files (.pkl)
        'label_encoders': {
            'label_encoder_bicepcurlphase.pkl': 'label_encoder_bicepcurlphase.pkl',
            'label_encoder_orientation.pkl': 'label_encoder_orientation.pkl',
            'label_encoder.pkl': 'label_encoder.pkl',
            'encoder.pkl': 'encoder.pkl'
        },
        
        # Raw data files (.csv)
        'raw_data': {
            'keypoints_data.csv': 'keypoints_data.csv',
            'keypoints_data1.csv': 'keypoints_data1.csv',
            'keypoints_data2.csv': 'keypoints_data2.csv',
            'keypoints_data3(all vids).csv': 'keypoints_data3_all_vids.csv',
            'keypointsfinal.csv': 'keypointsfinal.csv',
            'keypqqoints_data.csv': 'keypqqoints_data.csv'
        },
        
        # Processed data files (.csv)
        'processed_data': {
            'correct_processed.csv': 'correct_processed.csv',
            'incorrect_processed.csv': 'incorrect_processed.csv'
        }
    }
    
    # Move files to appropriate directories
    moved_files = []
    missing_files = []
    error_files = []
    
    for directory, files in file_mappings.items():
        for old_name, new_name in files.items():
            source_path = Path(old_name)
            dest_path = directories[directory] / new_name
            
            if source_path.exists():
                try:
                    # Move file to appropriate directory
                    shutil.move(str(source_path), str(dest_path))
                    print(f"✓ Moved: {old_name} -> {dest_path}")
                    moved_files.append(f"{directory}/{new_name}")
                except Exception as e:
                    print(f"❌ Error moving {old_name}: {e}")
                    error_files.append(old_name)
            else:
                print(f"⚠️  File not found: {old_name}")
                missing_files.append(old_name)
    
    # Move TensorFlow Lite model
    tflite_source = Path("movenet_singlepose_lightning.tflite")
    if tflite_source.exists():
        tflite_dest = directories['models'] / "movenet_singlepose_lightning.tflite"
        try:
            shutil.move(str(tflite_source), str(tflite_dest))
            print(f"✓ Moved: movenet_singlepose_lightning.tflite -> models/")
            moved_files.append("models/movenet_singlepose_lightning.tflite")
        except Exception as e:
            print(f"❌ Error moving TensorFlow Lite model: {e}")
    
    # Move other files to temp directory
    other_files = [
        'test.py',
        'main.py',
        'SVM.py',
        'MovenetModule.py',
        'AITrainerProject.py',
        'fyp.ipynb',
        'qodana.yaml',
        'pyvenv.cfg',
        'nolt.gif'
    ]
    
    for file_name in other_files:
        file_path = Path(file_name)
        if file_path.exists():
            try:
                temp_dest = directories['temp'] / file_name
                shutil.move(str(file_path), str(temp_dest))
                print(f"✓ Moved to temp/: {file_name}")
            except Exception as e:
                print(f"❌ Error moving {file_name}: {e}")
    
    # Create summary
    print("\n" + "=" * 60)
    print("ORGANIZATION SUMMARY")
    print("=" * 60)
    print(f"✓ Successfully moved {len(moved_files)} files")
    
    if missing_files:
        print(f"⚠️  {len(missing_files)} files were not found:")
        for file in missing_files:
            print(f"   - {file}")
    
    if error_files:
        print(f"❌ {len(error_files)} files had errors:")
        for file in error_files:
            print(f"   - {file}")
    
    # Update configuration files
    update_configuration_files(directories)
    
    print("\n" + "=" * 60)
    print("✅ File organization completed!")
    print("=" * 60)
    print("\nFiles are now organized as follows:")
    print("- TensorFlow models: models/tensorflow/")
    print("- Keras models: models/keras/")
    print("- Label encoders: models/label_encoders/")
    print("- Raw data: data/raw/")
    print("- Processed data: data/processed/")
    print("- Other files: temp/")

def update_configuration_files(directories):
    """
    Update configuration files to reflect new file paths.
    """
    print("\nUpdating configuration files...")
    
    # Update config/settings.py
    config_file = Path("config/settings.py")
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                content = f.read()
            
            # Update model paths
            content = content.replace(
                "'custom_ann': 'custom_ann_model.h5'",
                "'custom_ann': 'models/keras/custom_ann_model.h5'"
            )
            content = content.replace(
                "'label_encoder_bicepcurl': 'label_encoder_bicepcurlphase.pkl'",
                "'label_encoder_bicepcurl': 'models/label_encoders/label_encoder_bicepcurlphase.pkl'"
            )
            content = content.replace(
                "'label_encoder_orientation': 'label_encoder_orientation.pkl'",
                "'label_encoder_orientation': 'models/label_encoders/label_encoder_orientation.pkl'"
            )
            content = content.replace(
                "'encoder': 'encoder.pkl'",
                "'encoder': 'models/label_encoders/encoder.pkl'"
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
                "default='models/keras/custom_ann_model.h5'"
            )
            content = content.replace(
                "'label_encoder_bicepcurl': 'label_encoder_bicepcurlphase.pkl'",
                "'label_encoder_bicepcurl': 'models/label_encoders/label_encoder_bicepcurlphase.pkl'"
            )
            content = content.replace(
                "'label_encoder_orientation': 'label_encoder_orientation.pkl'",
                "'label_encoder_orientation': 'models/label_encoders/label_encoder_orientation.pkl'"
            )
            content = content.replace(
                "print('- custom_ann_model.h5')",
                "print('- models/keras/custom_ann_model.h5')"
            )
            content = content.replace(
                "print('- label_encoder_bicepcurlphase.pkl')",
                "print('- models/label_encoders/label_encoder_bicepcurlphase.pkl')"
            )
            content = content.replace(
                "print('- label_encoder_orientation.pkl')",
                "print('- models/label_encoders/label_encoder_orientation.pkl')"
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
                "'models/keras/custom_ann_model.h5'"
            )
            content = content.replace(
                "'label_encoder_bicepcurlphase.pkl'",
                "'models/label_encoders/label_encoder_bicepcurlphase.pkl'"
            )
            content = content.replace(
                "'label_encoder_orientation.pkl'",
                "'models/label_encoders/label_encoder_orientation.pkl'"
            )
            
            with open(setup_file, 'w') as f:
                f.write(content)
            
            print("✓ Updated setup.py")
        except Exception as e:
            print(f"❌ Error updating setup.py: {e}")

def create_data_summary():
    """
    Create a summary of the data files and their purposes.
    """
    print("\nCreating data summary...")
    
    data_summary = """# Data Files Summary

## Raw Data Files (data/raw/)
- **keypoints_data.csv**: Basic keypoint data
- **keypoints_data1.csv**: Extended keypoint dataset
- **keypoints_data2.csv**: Additional keypoint data
- **keypoints_data3_all_vids.csv**: Comprehensive dataset from all videos
- **keypointsfinal.csv**: Final processed keypoint data
- **keypqqoints_data.csv**: Additional keypoint dataset

## Processed Data Files (data/processed/)
- **correct_processed.csv**: Data with correct form labels
- **incorrect_processed.csv**: Data with incorrect form labels

## Model Files (models/)
### TensorFlow Models (models/tensorflow/)
- **saved_model.pb**: Main TensorFlow model
- **saved_model_backup.pb**: Backup TensorFlow model
- **keras_metadata.pb**: Keras model metadata
- **fingerprint.pb**: Model fingerprint
- **variables.index**: Model variables index
- **variables.data-00000-of-00001**: Model variables data

### Keras Models (models/keras/)
- **custom_ann_model.h5**: Custom Artificial Neural Network model
- **neural_network_model.h5**: Additional neural network model
- **model_weights.h5**: Model weights file

### Label Encoders (models/label_encoders/)
- **label_encoder_bicepcurlphase.pkl**: Encoder for bicep curl phases
- **label_encoder_orientation.pkl**: Encoder for body orientations
- **label_encoder.pkl**: General label encoder
- **encoder.pkl**: Additional encoder

### MoveNet Model
- **movenet_singlepose_lightning.tflite**: TensorFlow Lite pose detection model

## File Organization Benefits
- ✅ Cleaner project structure
- ✅ Easier to find specific files
- ✅ Better separation of concerns
- ✅ Improved maintainability
"""
    
    with open("data/README.md", 'w') as f:
        f.write(data_summary)
    
    print("✓ Created data/README.md with file descriptions")

if __name__ == "__main__":
    try:
        # Organize files
        organize_files()
        
        # Create data summary
        create_data_summary()
        
        print("\n🎉 File organization completed!")
        print("The project now has a clean, organized structure.")
        print("Check the data/README.md file for detailed descriptions.")
        
    except Exception as e:
        print(f"❌ Error during organization: {e}")
        import traceback
        traceback.print_exc() 