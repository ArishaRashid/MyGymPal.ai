"""
Custom Classifier Module

This module contains the CustomANN (Artificial Neural Network) classifier
for exercise form analysis and posture evaluation. It provides advanced
machine learning capabilities for real-time fitness assessment.

Author: MyGymPal.ai Team
Date: 2024
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import math
import warnings
from typing import Tuple, List, Optional, Dict, Any
from sklearn.preprocessing import LabelEncoder

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class CustomANN(tf.keras.Model):
    """
    Custom Artificial Neural Network for exercise form classification.
    
    This class implements a custom neural network architecture specifically
    designed for pose-based exercise form analysis and classification.
    """
    
    def __init__(self, input_dim: int = 64, hidden_dims: List[int] = [128, 64, 32], 
                 output_dim: int = 2, dropout_rate: float = 0.3):
        """
        Initialize the CustomANN model.
        
        Args:
            input_dim (int): Number of input features
            hidden_dims (List[int]): List of hidden layer dimensions
            output_dim (int): Number of output classes
            dropout_rate (float): Dropout rate for regularization
        """
        super(CustomANN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Build the neural network layers
        self.layers_list = []
        
        # Input layer
        self.layers_list.append(tf.keras.layers.Dense(hidden_dims[0], activation='relu', input_shape=(input_dim,)))
        self.layers_list.append(tf.keras.layers.Dropout(dropout_rate))
        
        # Hidden layers
        for dim in hidden_dims[1:]:
            self.layers_list.append(tf.keras.layers.Dense(dim, activation='relu'))
            self.layers_list.append(tf.keras.layers.Dropout(dropout_rate))
        
        # Output layer
        self.layers_list.append(tf.keras.layers.Dense(output_dim, activation='softmax'))
    
    def call(self, inputs):
        """
        Forward pass through the network.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Output predictions
        """
        x = inputs
        for layer in self.layers_list:
            x = layer(x)
        return x


class CustomClassifier:
    """
    A comprehensive classifier for exercise form analysis using CustomANN.
    
    This class provides functionality for:
    - Loading and managing CustomANN models
    - Preprocessing pose data for classification
    - Extracting features from keypoints
    - Analyzing exercise form and providing feedback
    """
    
    def __init__(self, model_path: str = "custom_ann_model.h5"):
        """
        Initialize the CustomANN classifier.
        
        Args:
            model_path (str): Path to the trained CustomANN model file
        """
        self.model_path = model_path
        self.model = None
        self.label_encoder_bicepcurl = None
        self.label_encoder_orientation = None
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load the CustomANN model and label encoders.
        
        Raises:
            FileNotFoundError: If model files are not found
            RuntimeError: If model loading fails
        """
        try:
            # Load CustomANN model
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✓ CustomANN model loaded successfully from {self.model_path}")
            
            # Load label encoders
            with open("label_encoder_bicepcurlphase.pkl", 'rb') as file:
                self.label_encoder_bicepcurl = pickle.load(file)
            
            with open("label_encoder_orientation.pkl", 'rb') as file:
                self.label_encoder_orientation = pickle.load(file)
                
            print("✓ Label encoders loaded successfully")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model file not found: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def preprocess_instance(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, bool, bool]:
        """
        Preprocess a new instance for classification.
        
        Args:
            df (pd.DataFrame): DataFrame containing keypoint data
            
        Returns:
            Tuple[pd.DataFrame, bool, bool]: Processed data, shoulders status, head status
        """
        # Create a copy to avoid modifying original data
        new_instance = pd.DataFrame(df)
        
        # Define keypoint column mapping
        column_mapping = {
            'keypoint_3': 'right_ear',
            'keypoint_4': 'left_ear',
            'keypoint_5': 'right_shoulder',
            'keypoint_6': 'left_shoulder',
            'keypoint_7': 'right_elbow',
            'keypoint_8': 'left_elbow',
            'keypoint_9': 'right_hand',
            'keypoint_10': 'left_hand',
            'keypoint_11': 'waist_right',
            'keypoint_12': 'waist_left',
            'keypoint_13': 'right_knee',
            'keypoint_14': 'left_knee',
            'keypoint_15': 'right_foot',
            'keypoint_16': 'left_foot'
        }
        
        # Handle missing values
        new_instance.fillna(-1, inplace=True)
        new_instance.replace('', -1, inplace=True)
        
        # Apply column renaming
        new_instance.rename(columns=column_mapping, inplace=True)
        
        # Split coordinates for all keypoints
        keypoint_columns = ['right_ear', 'left_ear', 'right_shoulder', 'left_shoulder',
                          'right_elbow', 'left_elbow', 'right_hand', 'left_hand',
                          'waist_right', 'waist_left', 'right_knee', 'left_knee',
                          'right_foot', 'left_foot']
        
        for column in keypoint_columns:
            self._split_coordinates(new_instance, column)
            self._clean_coordinates(new_instance, column)
        
        # Convert to numeric
        new_instance = new_instance.apply(pd.to_numeric, errors='coerce')
        
        # Handle missing foot positions
        self._handle_missing_foot_positions(new_instance)
        
        # Handle missing knee positions
        self._handle_missing_knee_positions(new_instance)
        
        # Calculate cumulative body size for normalization
        new_instance['cumulative_size'] = self._calculate_cumulative_size(new_instance)
        
        # Normalize keypoints
        self._normalize_keypoints(new_instance)
        
        # Determine orientation
        new_instance['orientation'] = new_instance.apply(self._determine_orientation, axis=1)
        
        # Check posture alignment
        shoulders_straight = self._is_shoulders_straight(new_instance)
        head_straight = self._is_head_straight(new_instance)
        
        if not shoulders_straight:
            return new_instance, False, True
        if not head_straight:
            return new_instance, True, False
        
        # Calculate angles
        self._calculate_angles(new_instance)
        
        # Calculate distances
        self._calculate_distances(new_instance)
        
        # Extract features
        new_instance = self._extract_features(new_instance)
        
        # Encode categorical variables
        new_instance['orientation_encoded'] = self.label_encoder_orientation.fit_transform(
            new_instance['orientation'])
        new_instance['bicep_curl_phase_encoded'] = self.label_encoder_bicepcurl.fit_transform(
            new_instance['bicep_curl_phase'])
        
        # Drop unnecessary columns
        new_instance.drop(["orientation", "bicep_curl_phase"], axis=1, inplace=True)
        
        # Drop coordinate columns
        columns_to_drop = [col for col in new_instance.columns if 'x' in col or 'y' in col]
        new_instance = new_instance.drop(columns=columns_to_drop)
        
        # Drop original keypoint columns
        new_instance.drop(columns=keypoint_columns, inplace=True)
        
        return new_instance, shoulders_straight, head_straight
    
    def predict_correctness(self, df: pd.DataFrame, shoulders: bool, head: bool) -> int:
        """
        Predict exercise correctness using the CustomANN model.
        
        Args:
            df (pd.DataFrame): Preprocessed feature data
            shoulders (bool): Whether shoulders are straight
            head (bool): Whether head is straight
            
        Returns:
            int: Prediction (0 for correct, 1 for incorrect)
        """
        if not shoulders or not head:
            return 1  # Incorrect posture
        
        # Make prediction
        prediction = self.model.predict(df)
        flattened_array = prediction.flatten()
        rounded_array = np.round(flattened_array).astype(int)
        
        return rounded_array[0] if len(rounded_array) > 0 else 1
    
    def check_posture_conditions(self, df: pd.DataFrame, shoulders: bool, head: bool) -> str:
        """
        Generate feedback based on posture conditions.
        
        Args:
            df (pd.DataFrame): Feature data
            shoulders (bool): Shoulder alignment status
            head (bool): Head alignment status
            
        Returns:
            str: Feedback message
        """
        posture_issues = []
        
        # Check knee angles
        if len(df) > 0:
            left_knee_angle = float(df['left_knee_angle'].iloc[0])
            right_knee_angle = float(df['right_knee_angle'].iloc[0])
            
            if left_knee_angle < 90 or right_knee_angle > 300:
                posture_issues.append("Make sure your knees are straight\n")
        
        # Check shoulder alignment
        if not shoulders:
            posture_issues.append("Keep your shoulders straight.\n")
        
        # Check head alignment
        if not head:
            posture_issues.append("Make sure that your head is not tilted.\n")
        
        return "".join(posture_issues)
    
    def _split_coordinates(self, df: pd.DataFrame, column_name: str) -> None:
        """
        Split coordinate tuples into separate x and y columns.
        
        Args:
            df (pd.DataFrame): DataFrame to modify
            column_name (str): Name of the coordinate column
        """
        new_columns = [f'{column_name}_x', f'{column_name}_y']
        
        if df[column_name].apply(lambda x: isinstance(x, tuple)).all():
            df[new_columns] = pd.DataFrame(df[column_name].tolist(), columns=new_columns)
        else:
            if new_columns[0] not in df.columns:
                df[new_columns] = df[column_name].apply(lambda x: pd.Series(x.split(' ')))
    
    def _clean_coordinates(self, df: pd.DataFrame, column_name: str) -> None:
        """
        Clean coordinate values by removing brackets and commas.
        
        Args:
            df (pd.DataFrame): DataFrame to modify
            column_name (str): Name of the coordinate column
        """
        for coordinate_type in ['_x', '_y']:
            df[column_name] = df[column_name].astype(str)
            df[column_name + coordinate_type] = df[column_name + coordinate_type].astype(str).str.replace(r'[()]', '')
            df[column_name + coordinate_type] = df[column_name + coordinate_type].astype(str).str.replace(',', '')
    
    def _handle_missing_foot_positions(self, df: pd.DataFrame) -> None:
        """
        Handle missing foot positions by estimating from knee positions.
        
        Args:
            df (pd.DataFrame): DataFrame to modify
        """
        df['right_foot_x'] = np.where(df['right_foot_x'].isnull(), df['right_knee_x'], df['right_foot_x'])
        df['right_foot_y'] = np.where(df['right_foot_y'].isnull(), df['right_knee_y'] + 100, df['right_foot_y'])
        df['left_foot_x'] = np.where(df['left_foot_x'].isnull(), df['left_knee_x'], df['left_foot_x'])
        df['left_foot_y'] = np.where(df['left_foot_y'].isnull(), df['left_knee_y'] + 100, df['left_foot_y'])
    
    def _handle_missing_knee_positions(self, df: pd.DataFrame) -> None:
        """
        Handle missing knee positions by estimating from waist positions.
        
        Args:
            df (pd.DataFrame): DataFrame to modify
        """
        df['right_knee_x'] = np.where(df['right_knee_x'].isnull(), df['waist_right_x'] + 30, df['right_knee_x'])
        df['right_knee_y'] = np.where(df['right_knee_y'].isnull(), df['waist_right_y'] + 50, df['right_knee_y'])
        df['left_knee_x'] = np.where(df['left_knee_x'].isnull(), df['waist_left_x'] - 30, df['left_knee_x'])
        df['left_knee_y'] = np.where(df['left_knee_y'].isnull(), df['waist_left_y'] + 50, df['left_knee_y'])
    
    def _calculate_cumulative_size(self, df: pd.DataFrame) -> List[float]:
        """
        Calculate cumulative body size for normalization.
        
        Args:
            df (pd.DataFrame): DataFrame with keypoint data
            
        Returns:
            List[float]: Cumulative size values
        """
        cumulative_size = []
        cumulative_distance = 0
        
        for index, row in df.iterrows():
            shoulder_waist_dist = self._euclidean_distance(
                row['right_shoulder_x'], row['right_shoulder_y'],
                row['waist_right_x'], row['waist_right_y'])
            waist_knees_dist = self._euclidean_distance(
                row['waist_right_x'], row['waist_right_y'],
                row['right_knee_x'], row['right_knee_y'])
            knees_feet_dist = self._euclidean_distance(
                row['right_knee_x'], row['right_knee_y'],
                row['right_knee_x'], row['right_knee_y'])
            
            cumulative_distance += shoulder_waist_dist + waist_knees_dist + knees_feet_dist
            cumulative_size.append(cumulative_distance)
        
        return cumulative_size
    
    def _normalize_keypoints(self, df: pd.DataFrame) -> None:
        """
        Normalize keypoints based on cumulative body size.
        
        Args:
            df (pd.DataFrame): DataFrame to modify
        """
        keypoints_columns = df.columns[3:-1]  # Skip metadata columns
        for column in keypoints_columns:
            df[column] = df[column] / df['cumulative_size']
        
        df.drop('cumulative_size', axis=1, inplace=True)
    
    def _determine_orientation(self, row: pd.Series) -> str:
        """
        Determine the orientation of the person in the frame.
        
        Args:
            row (pd.Series): Row containing keypoint data
            
        Returns:
            str: Orientation classification
        """
        # Calculate average x-coordinates
        avg_shoulder_x = (row['right_shoulder_x'] + row['left_shoulder_x']) / 2
        avg_waist_x = (row['right_waist_x'] + row['left_waist_x']) / 2
        avg_knee_x = (row['right_knee_x'] + row['left_knee_x']) / 2
        
        # Classify orientation
        if abs(avg_shoulder_x - avg_waist_x) < 20 and abs(avg_waist_x - avg_knee_x) < 20:
            return 'Front-Facing'
        elif avg_shoulder_x > avg_waist_x and avg_waist_x > avg_knee_x:
            return 'Right-Facing'
        elif avg_shoulder_x < avg_waist_x and avg_waist_x < avg_knee_x:
            return 'Left-Facing'
        else:
            return 'Unknown'
    
    def _is_shoulders_straight(self, df: pd.DataFrame, tolerance: float = 95) -> bool:
        """
        Check if shoulders are straight.
        
        Args:
            df (pd.DataFrame): DataFrame with keypoint data
            tolerance (float): Tolerance angle in degrees
            
        Returns:
            bool: True if shoulders are straight
        """
        if len(df) == 0:
            return False
        
        dy = df['left_shoulder_y'].iloc[0] - df['right_shoulder_y'].iloc[0]
        dx = df['left_shoulder_x'].iloc[0] - df['right_shoulder_x'].iloc[0]
        
        angle = math.atan2(dy, dx)
        angle_deg = math.degrees(angle)
        
        return abs(angle_deg) <= tolerance
    
    def _is_head_straight(self, df: pd.DataFrame, tolerance: float = 95) -> bool:
        """
        Check if head is straight.
        
        Args:
            df (pd.DataFrame): DataFrame with keypoint data
            tolerance (float): Tolerance angle in degrees
            
        Returns:
            bool: True if head is straight
        """
        if len(df) == 0:
            return False
        
        dy = df['left_ear_y'].iloc[0] - df['right_ear_y'].iloc[0]
        dx = df['left_ear_x'].iloc[0] - df['right_ear_x'].iloc[0]
        
        angle = math.atan2(dy, dx)
        angle_deg = math.degrees(angle)
        
        return abs(angle_deg) <= tolerance
    
    def _calculate_angles(self, df: pd.DataFrame) -> None:
        """
        Calculate various angles from keypoints.
        
        Args:
            df (pd.DataFrame): DataFrame to modify
        """
        angle_calculations = [
            ('right_elbow_angle', 'right_shoulder', 'right_elbow', 'right_hand'),
            ('left_elbow_angle', 'left_shoulder', 'left_elbow', 'left_hand'),
            ('right_waist_angle', 'right_shoulder', 'waist_right', 'right_knee'),
            ('left_waist_angle', 'left_shoulder', 'waist_left', 'left_knee'),
            ('right_knee_angle', 'right_knee', 'waist_right', 'right_foot'),
            ('left_knee_angle', 'left_knee', 'waist_left', 'left_foot'),
            ('back_angle', 'waist_right', 'waist_left', 'left_shoulder'),
            ('left_armpits_angle', 'left_elbow', 'left_shoulder', 'waist_right'),
            ('right_armpits_angle', 'right_elbow', 'right_shoulder', 'waist_left'),
            ('neck_angle', 'right_ear', 'right_shoulder', 'left_shoulder')
        ]
        
        for angle_name, p1, p2, p3 in angle_calculations:
            df[angle_name] = df.apply(lambda row: self._find_angle(row, p1, p2, p3), axis=1)
    
    def _calculate_distances(self, df: pd.DataFrame) -> None:
        """
        Calculate Euclidean distances between keypoints.
        
        Args:
            df (pd.DataFrame): DataFrame to modify
        """
        distance_calculations = [
            ('ear_shoulder_distance', 'right_shoulder_x', 'right_shoulder_y', 'right_ear_x', 'right_ear_y'),
            ('right_hands_elbows_distance', 'right_hand_x', 'right_hand_y', 'right_elbow_x', 'right_elbow_y'),
            ('left_hands_elbows_distance', 'left_hand_x', 'left_hand_y', 'left_elbow_x', 'left_elbow_y'),
            ('right_hands_shoulders_distance', 'right_hand_x', 'right_hand_y', 'right_shoulder_x', 'right_shoulder_y'),
            ('left_hands_shoulders_distance', 'left_hand_x', 'left_hand_y', 'left_shoulder_x', 'left_shoulder_y')
        ]
        
        for dist_name, x1_col, y1_col, x2_col, y2_col in distance_calculations:
            df[dist_name] = df.apply(lambda row: self._euclidean_distance(
                row[x1_col], row[y1_col], row[x2_col], row[y2_col]), axis=1)
    
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract exercise phase features from the data.
        
        Args:
            df (pd.DataFrame): DataFrame to modify
            
        Returns:
            pd.DataFrame: DataFrame with extracted features
        """
        # Define phase thresholds
        curl_up_threshold = 60
        curl_down_threshold = 120
        
        # Initialize phase column
        df['bicep_curl_phase'] = ''
        
        # Group by video and process each group
        for video_name, video_df in df.groupby('video_name'):
            video_df = video_df.sort_values(by='time_interval')
            
            # Identify phases based on elbow angles
            curl_up_indices = video_df[video_df['left_elbow_angle'] <= curl_up_threshold].index
            curl_down_indices = video_df[video_df['left_elbow_angle'] >= curl_down_threshold].index
            
            # Label phases
            df.loc[curl_up_indices, 'bicep_curl_phase'] = 'Curl Up'
            df.loc[curl_down_indices, 'bicep_curl_phase'] = 'Curl Down'
        
        # Encode video names
        encoder = LabelEncoder()
        df['video_name'] = df['video_name'].astype(str)
        df['video_name'] = encoder.fit_transform(df['video_name'])
        
        return df
    
    def _find_angle(self, row: pd.Series, shoulder: str, elbow: str, hand: str) -> float:
        """
        Calculate angle between three points.
        
        Args:
            row (pd.Series): Row containing coordinate data
            shoulder (str): Shoulder keypoint prefix
            elbow (str): Elbow keypoint prefix
            hand (str): Hand keypoint prefix
            
        Returns:
            float: Angle in degrees
        """
        x1, y1 = row[f'{shoulder}_x'], row[f'{shoulder}_y']
        x2, y2 = row[f'{elbow}_x'], row[f'{elbow}_y']
        x3, y3 = row[f'{hand}_x'], row[f'{hand}_y']
        
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        
        if angle < 0:
            angle += 360
        
        if 180 <= angle < 360:
            angle = 360 - angle
        
        return angle
    
    def _euclidean_distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Args:
            x1, y1 (float): First point coordinates
            x2, y2 (float): Second point coordinates
            
        Returns:
            float: Euclidean distance
        """
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Legacy functions for backward compatibility
def preprocess_new_instance(new_instance: pd.DataFrame, label_encoder_bicepcurl, 
                          label_encoder_orientation) -> Tuple[pd.DataFrame, bool, bool]:
    """
    Legacy function for preprocessing new instances.
    
    Args:
        new_instance (pd.DataFrame): Input data
        label_encoder_bicepcurl: Label encoder for bicep curl phases
        label_encoder_orientation: Label encoder for orientations
        
    Returns:
        Tuple[pd.DataFrame, bool, bool]: Processed data and status flags
    """
    classifier = CustomClassifier()
    return classifier.preprocess_instance(new_instance)


def predict_correctness(df: pd.DataFrame, shoulders: bool, head: bool, model) -> int:
    """
    Legacy function for predicting correctness.
    
    Args:
        df (pd.DataFrame): Feature data
        shoulders (bool): Shoulder alignment status
        head (bool): Head alignment status
        model: CustomANN model
        
    Returns:
        int: Prediction result
    """
    classifier = CustomClassifier()
    return classifier.predict_correctness(df, shoulders, head)


def check_posture_conditions(row: pd.DataFrame, shoulders: bool, head: bool) -> str:
    """
    Legacy function for checking posture conditions.
    
    Args:
        row (pd.DataFrame): Feature data
        shoulders (bool): Shoulder alignment status
        head (bool): Head alignment status
        
    Returns:
        str: Feedback message
    """
    classifier = CustomClassifier()
    return classifier.check_posture_conditions(row, shoulders, head) 