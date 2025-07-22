#!/usr/bin/env python3
"""
MyGymPal.ai - Entry Point Script

This script provides a simple way to run the MyGymPal.ai fitness application.
It handles imports and provides a clean interface for users.

Author: MyGymPal.ai Team
Date: 2024
"""

import sys
import os
import argparse

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """
    Main entry point for the MyGymPal.ai application.
    """
    parser = argparse.ArgumentParser(
        description="MyGymPal.ai - AI-Powered Fitness Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    # Run with default settings
  python run.py --webcam 1        # Use specific webcam
  python run.py --output video.mp4 # Save output video
        """
    )
    
    parser.add_argument(
        '--webcam', 
        type=int, 
        default=0,
        help='Webcam device number (default: 0)'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='Output video file path'
    )
    
    parser.add_argument(
        '--movenet', 
        type=str, 
        default='movenet_singlepose_lightning.tflite',
        help='Path to MoveNet model file'
    )
    
    parser.add_argument(
        '--custom_ann', 
        type=str, 
        default='models/keras/custom_ann_model.h5',
        help='Path to CustomANN model file'
    )
    
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    try:
        # Import the main application
        from app.main_app import FitnessApp
        
        # Create model paths dictionary
        model_paths = {
            'movenet': args.movenet,
            'custom_ann': args.custom_ann,
            'label_encoder_bicepcurl': 'models/label_encoders/label_encoder_bicepcurlphase.pkl',
            'label_encoder_orientation': 'models/label_encoders/label_encoder_orientation.pkl'
        }
        
        # Initialize and run the application
        app = FitnessApp(model_paths)
        app.process_video(video_source=args.webcam, output_path=args.output)
        
    except ImportError as e:
        print(f"Error: Could not import required modules. {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)
        
    except FileNotFoundError as e:
        print(f"Error: Model file not found. {e}")
        print("Please ensure all model files are in the project directory:")
        print("- movenet_singlepose_lightning.tflite")
        print("- custom_ann_model.h5")
        print("- label_encoder_bicepcurlphase.pkl")
        print("- label_encoder_orientation.pkl")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 