"""
Main Application Module

This module contains the main application logic for the MyGymPal.ai fitness application.
It orchestrates pose detection, exercise analysis, and real-time feedback.

Author: MyGymPal.ai Team
Date: 2024
"""

import cv2
import numpy as np
import pandas as pd
import time
import os
import warnings
from typing import Optional, Tuple, Dict, Any
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.pose_detection import PoseDetector
from core.visualization import Visualizer
from ml.custom_classifier import CustomClassifier

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class FitnessApp:
    """
    Main fitness application class that orchestrates all components.
    
    This class provides:
    - Real-time pose detection and analysis
    - Exercise form evaluation and feedback
    - Rep counting and performance tracking
    - Video processing and visualization
    """
    
    def __init__(self, model_paths: Optional[Dict[str, str]] = None):
        """
        Initialize the fitness application.
        
        Args:
            model_paths (Optional[Dict[str, str]]): Dictionary of model file paths
        """
        # Default model paths
        default_paths = {
            'movenet': 'movenet_singlepose_lightning.tflite',
            'svm': 'svm_81.11.h5',
            'label_encoder_bicepcurl': 'label_encoder_bicepcurlphase.pkl',
            'label_encoder_orientation': 'label_encoder_orientation.pkl'
        }
        
        self.model_paths = model_paths or default_paths
        
        # Initialize components
        self.pose_detector = None
        self.custom_classifier = None
        self.visualizer = None
        self._initialize_components()
        
        # Exercise tracking variables
        self.bicep_count = 0
        self.prev_form = 'unknown'
        self.frame_number = 0
        
        # Performance tracking
        self.start_time = None
        self.fps_counter = 0
        self.fps_start_time = time.time()
    
    def _initialize_components(self) -> None:
        """
        Initialize all application components.
        
        Raises:
            RuntimeError: If component initialization fails
        """
        try:
            print("Initializing MyGymPal.ai components...")
            
            # Initialize pose detector
            self.pose_detector = PoseDetector(self.model_paths['movenet'])
            print("✓ Pose detector initialized")
            
             # Initialize CustomANN classifier
            self.custom_classifier = CustomClassifier(self.model_paths['custom_ann'])
            print("✓ CustomANN classifier initialized")
            
            # Initialize visualizer
            self.visualizer = Visualizer()
            print("✓ Visualizer initialized")
            
            print("✓ All components initialized successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize components: {str(e)}")
    
    def process_video(self, video_source: int = 0, output_path: Optional[str] = None) -> None:
        """
        Process video stream for real-time exercise analysis.
        
        Args:
            video_source (int): Video source (0 for webcam, or file path)
            output_path (Optional[str]): Path to save output video
        """
        # Initialize video capture
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return
        
        # Get video properties
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        fps = int(cap.get(5))
        
        # Initialize video writer if output path is provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        print(f"Starting video processing...")
        print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}")
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("Error reading frame. Exiting...")
                    break
                
                if frame.size == 0:
                    print("Empty frame. Skipping...")
                    continue
                
                # Process frame
                processed_frame = self._process_frame(frame)
                
                # Display frame
                cv2.imshow('MyGymPal.ai - Real-time Exercise Analysis', processed_frame)
                
                # Save frame if output path is provided
                if out:
                    out.write(processed_frame)
                
                # Check for exit key
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            print("Video processing completed")
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame for exercise analysis.
        
        Args:
            frame (np.ndarray): Input frame
            
        Returns:
            np.ndarray: Processed frame with annotations
        """
        # Resize frame for processing
        frame = cv2.resize(frame, (640, 480))
        
        # Track processing time
        start_time = time.time()
        
        # Detect pose
        keypoints = self.pose_detector.detect_pose(frame)
        
        # Update frame counter
        self.frame_number += 1
        
        # Create DataFrame for analysis
        df = self._create_analysis_dataframe(keypoints)
        
        # Analyze exercise form
        correctness, feedback = self._analyze_exercise_form(df, keypoints)
        
        # Update exercise count
        self._update_exercise_count(keypoints)
        
        # Draw visualizations
        frame = self._draw_visualizations(frame, keypoints, correctness, feedback)
        
        # Calculate and display FPS
        processing_time = time.time() - start_time
        self._update_fps(processing_time)
        
        return frame
    
    def _create_analysis_dataframe(self, keypoints: np.ndarray) -> pd.DataFrame:
        """
        Create DataFrame for exercise analysis.
        
        Args:
            keypoints (np.ndarray): Keypoints from pose detection
            
        Returns:
            pd.DataFrame: DataFrame with keypoint data
        """
        # Extract keypoints 3-16 (body keypoints)
        keypoints_3_to_16 = [self.pose_detector.get_coordinates(keypoints, i)[:2] 
                            for i in range(3, 17)]
        
        # Create DataFrame
        data = {
            'video_name': 'webcam',
            'frame_number': self.frame_number,
            'time_interval': self.frame_number / 30  # Assuming 30 FPS
        }
        
        # Add keypoint data
        for i, keypoint in enumerate(keypoints_3_to_16):
            data[f'keypoint_{i+3}'] = keypoint
        
        return pd.DataFrame([data])
    
    def _analyze_exercise_form(self, df: pd.DataFrame, keypoints: np.ndarray) -> Tuple[int, str]:
        """
        Analyze exercise form using SVM classifier.
        
        Args:
            df (pd.DataFrame): Keypoint data
            keypoints (np.ndarray): Raw keypoints
            
        Returns:
            Tuple[int, str]: Correctness prediction and feedback
        """
        try:
            # Preprocess data
            df_processed, shoulders_straight, head_straight = self.custom_classifier.preprocess_instance(df)
            
            # Predict correctness
            correctness = self.custom_classifier.predict_correctness(df_processed, shoulders_straight, head_straight)
            
            # Generate feedback
            if correctness == 1:  # Incorrect form
                feedback = self.custom_classifier.check_posture_conditions(df_processed, shoulders_straight, head_straight)
            else:  # Correct form
                feedback = ""
            
            return correctness, feedback
            
        except Exception as e:
            print(f"Error in exercise form analysis: {str(e)}")
            return 1, "Analysis error"  # Default to incorrect
    
    def _update_exercise_count(self, keypoints: np.ndarray) -> None:
        """
        Update exercise count based on form changes.
        
        Args:
            keypoints (np.ndarray): Keypoints from pose detection
        """
        current_form = self.pose_detector.get_bicep_curl_form(keypoints)
        
        # Count bicep curls based on form transitions
        if self.prev_form == 'open' and current_form == 'closed':
            self.bicep_count += 0.5
        elif self.prev_form == 'closed' and current_form == 'open':
            self.bicep_count += 0.5
        
        self.prev_form = current_form
    
    def _draw_visualizations(self, frame: np.ndarray, keypoints: np.ndarray, 
                           correctness: int, feedback: str) -> np.ndarray:
        """
        Draw all visualizations on the frame.
        
        Args:
            frame (np.ndarray): Input frame
            keypoints (np.ndarray): Keypoints from pose detection
            correctness (int): Form correctness (0=correct, 1=incorrect)
            feedback (str): Feedback message
            
        Returns:
            np.ndarray: Frame with visualizations
        """
        # Draw keypoints and connections
        frame = self.pose_detector.draw_keypoints(frame, keypoints)
        frame = self.pose_detector.draw_connections(frame, keypoints)
        
        # Draw angles for exercise analysis
        angle_points = [
            (5, 7, 9),   # Right shoulder angle
            (6, 8, 10),  # Left shoulder angle
            (6, 12, 14), # Right leg angle
            (5, 11, 13), # Left leg angle
            (11, 13, 15), # Right knee angle
            (12, 14, 16)  # Left knee angle
        ]
        
        for p1, p2, p3 in angle_points:
            frame = self.visualizer.draw_angle(
                frame, p1, p2, p3, keypoints,
                correct_posture=(correctness == 0),
                feedback=feedback if correctness == 1 else ""
            )
        
        # Display exercise count
        frame = self.visualizer.display_bicep_count(frame, int(self.bicep_count))
        
        return frame
    
    def _update_fps(self, processing_time: float) -> None:
        """
        Update and display FPS information.
        
        Args:
            processing_time (float): Time taken to process current frame
        """
        self.fps_counter += 1
        
        if self.fps_counter % 30 == 0:  # Update every 30 frames
            current_time = time.time()
            elapsed_time = current_time - self.fps_start_time
            fps = self.fps_counter / elapsed_time
            print(f"FPS: {fps:.2f}, Latency: {processing_time:.3f}s")
    
    def get_exercise_stats(self) -> Dict[str, Any]:
        """
        Get current exercise statistics.
        
        Returns:
            Dict[str, Any]: Dictionary containing exercise statistics
        """
        return {
            'bicep_count': int(self.bicep_count),
            'frame_number': self.frame_number,
            'current_form': self.prev_form
        }
    
    def reset_counters(self) -> None:
        """Reset all exercise counters."""
        self.bicep_count = 0
        self.frame_number = 0
        self.prev_form = 'unknown'
        print("Counters reset")


def main():
    """
    Main entry point for the fitness application.
    """
    print("=" * 50)
    print("MyGymPal.ai - AI-Powered Fitness Application")
    print("=" * 50)
    
    try:
        # Initialize application
        app = FitnessApp()
        
        # Start video processing
        print("\nStarting real-time exercise analysis...")
        print("Press 'q' to quit")
        print("-" * 50)
        
        app.process_video(video_source=0)  # Use webcam
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Application error: {str(e)}")
    finally:
        print("\nThank you for using MyGymPal.ai!")


if __name__ == "__main__":
    main() 