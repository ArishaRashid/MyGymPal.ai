"""
Pose Detection Module

This module handles real-time pose detection using MoveNet and provides
utilities for keypoint extraction, angle calculations, and pose analysis.

Author: MyGymPal.ai Team
Date: 2024
"""

import tensorflow as tf
import numpy as np
import cv2
import math
from typing import Tuple, List, Optional, Dict, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class PoseDetector:
    """
    A class for detecting human poses using MoveNet.
    
    This class provides functionality for:
    - Loading and managing MoveNet models
    - Processing video frames for pose detection
    - Extracting keypoints and calculating angles
    - Analyzing exercise form and posture
    """
    
    def __init__(self, model_path: str = 'movenet_singlepose_lightning.tflite'):
        """
        Initialize the pose detector with MoveNet model.
        
        Args:
            model_path (str): Path to the MoveNet TensorFlow Lite model file
        """
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._load_model()
        
        # Define body part connections for visualization
        self.EDGES = {
            (0, 1): 'm',   # nose to left_eye
            (0, 2): 'c',   # nose to right_eye
            (1, 3): 'm',   # left_eye to left_ear
            (2, 4): 'c',   # right_eye to right_ear
            (0, 5): 'm',   # nose to left_shoulder
            (0, 6): 'c',   # nose to right_shoulder
            (5, 7): 'm',   # left_shoulder to left_elbow
            (7, 9): 'm',   # left_elbow to left_wrist
            (6, 8): 'c',   # right_shoulder to right_elbow
            (8, 10): 'c',  # right_elbow to right_wrist
            (5, 6): 'y',   # left_shoulder to right_shoulder
            (5, 11): 'm',  # left_shoulder to left_hip
            (6, 12): 'c',  # right_shoulder to right_hip
            (11, 12): 'y', # left_hip to right_hip
            (11, 13): 'm', # left_hip to left_knee
            (13, 15): 'm', # left_knee to left_ankle
            (12, 14): 'c', # right_hip to right_knee
            (14, 16): 'c'  # right_knee to right_ankle
        }
    
    def _load_model(self) -> None:
        """
        Load the MoveNet TensorFlow Lite model and allocate tensors.
        
        Raises:
            FileNotFoundError: If the model file is not found
            RuntimeError: If model loading fails
        """
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"✓ MoveNet model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def detect_pose(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect pose in a single frame.
        
        Args:
            frame (np.ndarray): Input frame (BGR format)
            
        Returns:
            np.ndarray: Keypoints with scores [y, x, confidence]
        """
        # Resize and pad the image to 192x192
        img = tf.image.resize_with_pad(np.expand_dims(frame, axis=0), 192, 192)
        input_image = tf.cast(img, dtype=tf.float32)
        
        # Set input tensor and run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], np.array(input_image))
        self.interpreter.invoke()
        
        # Get output keypoints
        keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])
        return keypoints_with_scores
    
    def get_coordinates(self, keypoints: np.ndarray, point_number: int) -> Tuple[int, int, float]:
        """
        Extract coordinates of a specific keypoint.
        
        Args:
            keypoints (np.ndarray): Keypoints array from MoveNet
            point_number (int): Index of the keypoint (0-16)
            
        Returns:
            Tuple[int, int, float]: (y, x, confidence) coordinates
            
        Raises:
            IndexError: If point_number is out of range
        """
        if point_number < 0 or point_number >= 17:
            raise IndexError(f"Point number {point_number} is out of range (0-16)")
        
        # Reshape keypoints to match frame dimensions
        shaped = np.squeeze(np.multiply(keypoints, [480, 640, 1]))
        
        if point_number < len(shaped):
            y, x, confidence = shaped[point_number]
            return int(y), int(x), confidence
        else:
            return 0, 0, 0.0
    
    def calculate_angle(self, p1: int, p2: int, p3: int, keypoints: np.ndarray) -> float:
        """
        Calculate the angle between three keypoints.
        
        Args:
            p1 (int): First keypoint index
            p2 (int): Second keypoint index (vertex)
            p3 (int): Third keypoint index
            keypoints (np.ndarray): Keypoints array
            
        Returns:
            float: Angle in degrees (0-360)
        """
        y1, x1, c1 = self.get_coordinates(keypoints, p1)
        y2, x2, c2 = self.get_coordinates(keypoints, p2)
        y3, x3, c3 = self.get_coordinates(keypoints, p3)
        
        # Calculate angle using atan2
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        
        # Normalize angle to 0-360 range
        if angle < 0:
            angle = abs(angle)
        
        return angle
    
    def get_bicep_curl_form(self, keypoints: np.ndarray) -> str:
        """
        Determine the form of bicep curl exercise.
        
        Args:
            keypoints (np.ndarray): Keypoints from pose detection
            
        Returns:
            str: Form classification ('open', 'closed', or 'unknown')
        """
        right_shoulder_angle = self.calculate_angle(5, 7, 9, keypoints)
        left_shoulder_angle = self.calculate_angle(6, 8, 10, keypoints)
        
        # Classify based on shoulder angles
        if 140 <= right_shoulder_angle <= 200 or 140 <= left_shoulder_angle <= 200:
            return "open"
        elif 0 <= right_shoulder_angle <= 40 or 0 <= left_shoulder_angle <= 40:
            return "closed"
        else:
            return "unknown"
    
    def get_squat_form(self, keypoints: np.ndarray) -> str:
        """
        Determine the form of squat exercise.
        
        Args:
            keypoints (np.ndarray): Keypoints from pose detection
            
        Returns:
            str: Form classification ('squat', 'standing', or 'unknown')
        """
        left_waist_angle = self.calculate_angle(11, 12, 13, keypoints)
        right_waist_angle = self.calculate_angle(11, 12, 14, keypoints)
        left_knee_angle = self.calculate_angle(13, 14, 15, keypoints)
        right_knee_angle = self.calculate_angle(14, 13, 16, keypoints)
        
        # Classify based on waist and knee angles
        if (0 <= left_waist_angle <= 70 or 0 <= right_waist_angle <= 70) and \
           (0 <= left_knee_angle <= 70 or 0 <= right_knee_angle <= 70):
            return "squat"
        elif (110 <= left_waist_angle <= 180 or 110 <= right_waist_angle <= 180) and \
             (110 <= left_knee_angle <= 180 or 110 <= right_knee_angle <= 180):
            return "standing"
        else:
            return "unknown"
    
    def draw_keypoints(self, frame: np.ndarray, keypoints: np.ndarray, 
                      confidence_threshold: float = 0.3) -> np.ndarray:
        """
        Draw keypoints on the frame.
        
        Args:
            frame (np.ndarray): Input frame
            keypoints (np.ndarray): Keypoints array
            confidence_threshold (float): Minimum confidence for drawing
            
        Returns:
            np.ndarray: Frame with keypoints drawn
        """
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
        
        for kp in shaped[5:]:  # Skip face keypoints
            ky, kx, kp_conf = kp
            if kp_conf > confidence_threshold:
                cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)
        
        return frame
    
    def draw_connections(self, frame: np.ndarray, keypoints: np.ndarray, 
                        confidence_threshold: float = 0.3) -> np.ndarray:
        """
        Draw connections between keypoints on the frame.
        
        Args:
            frame (np.ndarray): Input frame
            keypoints (np.ndarray): Keypoints array
            confidence_threshold (float): Minimum confidence for drawing
            
        Returns:
            np.ndarray: Frame with connections drawn
        """
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
        
        for edge, color in self.EDGES.items():
            p1, p2 = edge
            if p1 > 5 and p2 > 5:  # Skip face connections
                y1, x1, c1 = shaped[p1]
                y2, x2, c2 = shaped[p2]
                
                if c1 > confidence_threshold and c2 > confidence_threshold:
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        
        return frame


def rescale_points(keypoints: np.ndarray, frame: np.ndarray) -> Tuple[int, int]:
    """
    Rescale keypoint coordinates to match frame dimensions.
    
    Args:
        keypoints (np.ndarray): Keypoints array
        frame (np.ndarray): Target frame
        
    Returns:
        Tuple[int, int]: Rescaled (x, y) coordinates
    """
    key_y, key_x = np.squeeze(keypoints)
    
    width = int(1)
    height = int(1)
    new_width, new_height = 1, 1
    
    key_x *= new_width / width
    key_y *= new_height / height
    
    return int(key_x), int(key_y)


def rescale_three_points(keypoints: np.ndarray, frame: np.ndarray) -> Tuple[int, int, int]:
    """
    Rescale keypoint coordinates with confidence to match frame dimensions.
    
    Args:
        keypoints (np.ndarray): Keypoints array with confidence
        frame (np.ndarray): Target frame
        
    Returns:
        Tuple[int, int, int]: Rescaled (x, y, confidence) coordinates
    """
    key_y, key_x, c = np.squeeze(keypoints)
    
    width = int(frame.shape[1])
    height = int(frame.shape[0])
    new_width, new_height = 640, 1136
    
    key_x *= new_width / width
    key_y *= new_height / height
    
    return int(key_x), int(key_y), int(c) 