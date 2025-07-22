"""
Configuration Settings Module

This module contains all configuration settings, constants, and parameters
for the MyGymPal.ai fitness application.

Author: MyGymPal.ai Team
Date: 2024
"""

import os
from typing import Dict, List, Tuple

# =============================================================================
# MODEL PATHS AND FILES
# =============================================================================

# Default model file paths
MODEL_PATHS = {
    'movenet': 'models/movenet_singlepose_lightning.tflite',
    'custom_ann': 'models/keras/custom_ann_model.h5',
    'label_encoder_bicepcurl': 'models/label_encoders/label_encoder_bicepcurlphase.pkl',
    'label_encoder_orientation': 'models/label_encoders/label_encoder_orientation.pkl',
    'encoder': 'models/label_encoders/encoder.pkl'
}

# Model file descriptions
MODEL_DESCRIPTIONS = {
    'movenet': 'MoveNet Lightning model for real-time pose detection',
    'custom_ann': 'CustomANN model for exercise form classification',
    'label_encoder_bicepcurl': 'Label encoder for bicep curl phases',
    'label_encoder_orientation': 'Label encoder for body orientations',
    'encoder': 'General label encoder for exercise types'
}

# =============================================================================
# POSE DETECTION SETTINGS
# =============================================================================

# MoveNet configuration
MOVENET_CONFIG = {
    'input_size': (192, 192),
    'confidence_threshold': 0.3,
    'keypoint_count': 17,
    'body_keypoints_start': 5,  # Skip face keypoints
    'body_keypoints_end': 17
}

# Keypoint indices for different body parts
KEYPOINT_INDICES = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Body part connections for visualization
BODY_CONNECTIONS = {
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

# =============================================================================
# EXERCISE ANALYSIS SETTINGS
# =============================================================================

# Bicep curl analysis parameters
BICEP_CURL_CONFIG = {
    'open_angle_range': (140, 200),
    'closed_angle_range': (0, 40),
    'curl_up_threshold': 60,
    'curl_down_threshold': 120,
    'shoulder_angle_points': [(5, 7, 9), (6, 8, 10)]  # (shoulder, elbow, wrist)
}

# Squat analysis parameters
SQUAT_CONFIG = {
    'squat_waist_angle_range': (0, 70),
    'squat_knee_angle_range': (0, 70),
    'standing_waist_angle_range': (110, 180),
    'standing_knee_angle_range': (110, 180),
    'waist_angle_points': [(11, 12, 13), (11, 12, 14)],  # (hip, hip, knee)
    'knee_angle_points': [(13, 14, 15), (14, 13, 16)]   # (knee, hip, ankle)
}

# Posture analysis parameters
POSTURE_CONFIG = {
    'shoulder_tolerance': 95,  # degrees
    'head_tolerance': 95,      # degrees
    'knee_straight_min': 90,   # degrees
    'knee_straight_max': 300   # degrees
}

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================

# Color definitions (BGR format)
COLORS = {
    'correct': (0, 255, 0),      # Green
    'incorrect': (0, 0, 255),    # Red
    'neutral': (255, 255, 255),  # White
    'warning': (0, 165, 255),    # Orange
    'info': (255, 255, 0),       # Cyan
    'count': (193, 111, 157),    # Purple
    'connection': (0, 0, 255),   # Red for connections
    'keypoint': (0, 255, 0)      # Green for keypoints
}

# Drawing parameters
DRAWING_CONFIG = {
    'line_thickness': 2,
    'circle_radius': 4,
    'font_scale': 1,
    'font_thickness': 2,
    'text_offset': (50, 50),
    'feedback_box_width': 500,
    'feedback_box_height_offset': 20
}

# Display settings
DISPLAY_CONFIG = {
    'default_resolution': (640, 480),
    'count_box_padding': 20,
    'count_box_radius': 10,
    'fps_update_interval': 30  # Update FPS every 30 frames
}

# =============================================================================
# VIDEO PROCESSING SETTINGS
# =============================================================================

# Video processing parameters
VIDEO_CONFIG = {
    'default_fps': 30,
    'frame_buffer_size': 100,
    'output_codec': 'mp4v',
    'output_extension': '.mp4'
}

# Camera settings
CAMERA_CONFIG = {
    'default_source': 0,
    'resolution': (640, 480),
    'fps': 30,
    'brightness': 0,
    'contrast': 0,
    'saturation': 0
}

# =============================================================================
# MACHINE LEARNING SETTINGS
# =============================================================================

# CustomANN model parameters
CUSTOM_ANN_CONFIG = {
    'input_dim': 64,
    'hidden_dims': [128, 64, 32],
    'output_dim': 2,
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100
}

# Feature extraction parameters
FEATURE_CONFIG = {
    'normalization_method': 'cumulative_size',
    'angle_calculation_method': 'atan2',
    'distance_method': 'euclidean'
}

# Data preprocessing parameters
PREPROCESSING_CONFIG = {
    'missing_value_fill': -1,
    'coordinate_cleaning': True,
    'angle_normalization': True,
    'feature_scaling': True
}

# =============================================================================
# PERFORMANCE SETTINGS
# =============================================================================

# Performance monitoring
PERFORMANCE_CONFIG = {
    'enable_fps_monitoring': True,
    'enable_latency_monitoring': True,
    'enable_memory_monitoring': False,
    'log_performance': True
}

# Memory management
MEMORY_CONFIG = {
    'max_frame_buffer': 1000,
    'garbage_collection_interval': 100,
    'cleanup_threshold': 0.8
}

# =============================================================================
# LOGGING AND DEBUGGING
# =============================================================================

# Logging configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_file': 'mygympal.log',
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'enable_console_logging': True,
    'enable_file_logging': True
}

# Debug settings
DEBUG_CONFIG = {
    'enable_debug_mode': False,
    'show_keypoints': True,
    'show_angles': True,
    'show_connections': True,
    'show_feedback': True,
    'show_count': True
}

# =============================================================================
# APPLICATION SETTINGS
# =============================================================================

# Application metadata
APP_CONFIG = {
    'name': 'MyGymPal.ai',
    'version': '1.0.0',
    'description': 'AI-Powered Fitness Application',
    'author': 'MyGymPal.ai Team',
    'contact': 'support@mygympal.ai'
}

# Supported exercises
SUPPORTED_EXERCISES = [
    'bicep_curl',
    'squat',
    'push_up',
    'plank',
    'lunge'
]

# Exercise configurations
EXERCISE_CONFIGS = {
    'bicep_curl': {
        'name': 'Bicep Curl',
        'description': 'Arm curl exercise for bicep development',
        'keypoints': ['shoulder', 'elbow', 'wrist'],
        'angles': ['shoulder_angle'],
        'phases': ['curl_up', 'curl_down']
    },
    'squat': {
        'name': 'Squat',
        'description': 'Lower body exercise for leg strength',
        'keypoints': ['hip', 'knee', 'ankle'],
        'angles': ['knee_angle', 'hip_angle'],
        'phases': ['down', 'up']
    }
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_model_path(model_name: str) -> str:
    """
    Get the path for a specific model file.
    
    Args:
        model_name (str): Name of the model
        
    Returns:
        str: Path to the model file
    """
    return MODEL_PATHS.get(model_name, '')

def get_color(color_name: str) -> Tuple[int, int, int]:
    """
    Get a color by name.
    
    Args:
        color_name (str): Name of the color
        
    Returns:
        Tuple[int, int, int]: BGR color tuple
    """
    return COLORS.get(color_name, (255, 255, 255))

def get_keypoint_index(body_part: str) -> int:
    """
    Get the keypoint index for a body part.
    
    Args:
        body_part (str): Name of the body part
        
    Returns:
        int: Keypoint index
    """
    return KEYPOINT_INDICES.get(body_part, -1)

def is_debug_enabled() -> bool:
    """
    Check if debug mode is enabled.
    
    Returns:
        bool: True if debug mode is enabled
    """
    return DEBUG_CONFIG.get('enable_debug_mode', False)

def get_supported_exercises() -> List[str]:
    """
    Get list of supported exercises.
    
    Returns:
        List[str]: List of supported exercise names
    """
    return SUPPORTED_EXERCISES.copy() 