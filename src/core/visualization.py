"""
Visualization Module

This module handles all visual elements including:
- Drawing angles and keypoints on frames
- Displaying exercise counts and feedback
- Rendering posture indicators and warnings

Author: MyGymPal.ai Team
Date: 2024
"""

import cv2
import numpy as np
import math
from typing import Tuple, Optional


class Visualizer:
    """
    A class for handling all visualization tasks in the fitness application.
    
    This class provides methods for:
    - Drawing angles and keypoints on video frames
    - Displaying exercise counts and performance metrics
    - Rendering feedback and posture warnings
    - Creating visual indicators for exercise form
    """
    
    def __init__(self):
        """Initialize the visualizer with default settings."""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.font_thickness = 2
        self.line_thickness = 2
        self.circle_radius = 4
        
        # Color definitions for different states
        self.colors = {
            'correct': (0, 255, 0),      # Green
            'incorrect': (0, 0, 255),    # Red
            'neutral': (255, 255, 255),  # White
            'warning': (0, 165, 255),    # Orange
            'info': (255, 255, 0),       # Cyan
            'count': (193, 111, 157)     # Purple
        }
    
    def draw_angle(self, frame: np.ndarray, p1: int, p2: int, p3: int, 
                   keypoints: np.ndarray, confidence_threshold: float = 0.3,
                   draw: bool = True, correct_posture: bool = True, 
                   feedback: str = "") -> np.ndarray:
        """
        Draw an angle between three keypoints on the frame.
        
        Args:
            frame (np.ndarray): Input frame to draw on
            p1, p2, p3 (int): Keypoint indices for angle calculation
            keypoints (np.ndarray): Keypoints array from pose detection
            confidence_threshold (float): Minimum confidence for drawing
            draw (bool): Whether to draw the angle
            correct_posture (bool): Whether the posture is correct
            feedback (str): Feedback text to display
            
        Returns:
            np.ndarray: Frame with angle drawn
        """
        # Get coordinates for the three points
        y1, x1, c1 = self._get_coordinates(keypoints, p1)
        y2, x2, c2 = self._get_coordinates(keypoints, p2)
        y3, x3, c3 = self._get_coordinates(keypoints, p3)
        
        # Calculate angle
        angle = self._calculate_angle(y1, x1, y2, x2, y3, x3)
        
        # Choose color based on posture correctness
        color = self.colors['correct'] if correct_posture else self.colors['incorrect']
        
        # Draw if confidence is sufficient
        if (c1 > confidence_threshold and c2 > confidence_threshold and c3 > confidence_threshold):
            if draw:
                # Draw lines connecting the points
                cv2.line(frame, (x1, y1), (x2, y2), color, self.line_thickness)
                cv2.line(frame, (x3, y3), (x2, y2), color, self.line_thickness)
                
                # Draw circles at keypoints
                cv2.circle(frame, (x1, y1), self.circle_radius, color, -1)
                cv2.circle(frame, (x1, y1), self.circle_radius + 3, color, 1)
                cv2.circle(frame, (x2, y2), self.circle_radius, color, -1)
                cv2.circle(frame, (x2, y2), self.circle_radius + 3, color, 1)
                cv2.circle(frame, (x3, y3), self.circle_radius, color, -1)
                cv2.circle(frame, (x3, y3), self.circle_radius + 3, color, 1)
                
                # Display angle value
                cv2.putText(frame, str(int(angle)), (x2 - 50, y2 + 50), 
                           cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
                
                # Display feedback if provided
                if feedback:
                    self._display_feedback(frame, feedback)
        
        return frame
    
    def display_exercise_count(self, frame: np.ndarray, count: int, 
                             exercise_type: str = "Bicep") -> np.ndarray:
        """
        Display exercise count on the frame.
        
        Args:
            frame (np.ndarray): Input frame
            count (int): Current exercise count
            exercise_type (str): Type of exercise being performed
            
        Returns:
            np.ndarray: Frame with count displayed
        """
        height, width, _ = frame.shape
        text = f"{exercise_type} Count: {int(count)}"
        
        # Calculate text size
        text_size = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)[0]
        text_width, text_height = text_size
        
        # Calculate box dimensions
        box_width = text_width + 20
        box_height = text_height + 20
        
        # Calculate box position (top-right corner)
        box_x = width - box_width - 10
        box_y = 10 + text_height
        
        # Draw rounded rectangle background
        self._draw_rounded_rectangle(frame, box_x, box_y - text_height, 
                                   box_x + box_width, box_y + 10, 
                                   self.colors['count'], radius=10)
        
        # Draw text
        cv2.putText(frame, text, (box_x + 10, box_y), self.font, 
                   self.font_scale, self.colors['neutral'], self.font_thickness)
        
        return frame
    
    def display_squat_count(self, frame: np.ndarray, count: int) -> np.ndarray:
        """
        Display squat count on the frame.
        
        Args:
            frame (np.ndarray): Input frame
            count (int): Current squat count
            
        Returns:
            np.ndarray: Frame with squat count displayed
        """
        return self.display_exercise_count(frame, count, "Squat")
    
    def display_bicep_count(self, frame: np.ndarray, count: int) -> np.ndarray:
        """
        Display bicep curl count on the frame.
        
        Args:
            frame (np.ndarray): Input frame
            count (int): Current bicep curl count
            
        Returns:
            np.ndarray: Frame with bicep count displayed
        """
        return self.display_exercise_count(frame, count, "Bicep")
    
    def _get_coordinates(self, keypoints: np.ndarray, point_number: int) -> Tuple[int, int, float]:
        """
        Extract coordinates of a specific keypoint.
        
        Args:
            keypoints (np.ndarray): Keypoints array
            point_number (int): Keypoint index
            
        Returns:
            Tuple[int, int, float]: (y, x, confidence) coordinates
        """
        shaped = np.squeeze(np.multiply(keypoints, [480, 640, 1]))
        
        if point_number < len(shaped):
            y, x, confidence = shaped[point_number]
            return int(y), int(x), confidence
        else:
            return 0, 0, 0.0
    
    def _calculate_angle(self, y1: int, x1: int, y2: int, x2: int, 
                        y3: int, x3: int) -> float:
        """
        Calculate angle between three points.
        
        Args:
            y1, x1: First point coordinates
            y2, x2: Second point coordinates (vertex)
            y3, x3: Third point coordinates
            
        Returns:
            float: Angle in degrees
        """
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle = abs(angle)
        return angle
    
    def _draw_rounded_rectangle(self, frame: np.ndarray, x1: int, y1: int, 
                               x2: int, y2: int, color: Tuple[int, int, int], 
                               radius: int = 10) -> None:
        """
        Draw a rounded rectangle on the frame.
        
        Args:
            frame (np.ndarray): Frame to draw on
            x1, y1, x2, y2 (int): Rectangle coordinates
            color (Tuple[int, int, int]): BGR color
            radius (int): Corner radius
        """
        # Draw main rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        
        # Draw rounded corners
        cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        
        # Draw corner circles
        cv2.circle(frame, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(frame, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(frame, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(frame, (x2 - radius, y2 - radius), radius, color, -1)
    
    def _display_feedback(self, frame: np.ndarray, feedback: str) -> None:
        """
        Display feedback text on the frame.
        
        Args:
            frame (np.ndarray): Frame to draw on
            feedback (str): Feedback text to display
        """
        if not feedback:
            return
        
        # Split feedback into lines
        feedback_lines = feedback.split('\n')
        y_position = 420  # Starting y-position
        
        if feedback_lines:
            # Calculate box dimensions
            max_line_length = max(len(line) for line in feedback_lines)
            text_width, text_height = cv2.getTextSize("W" * max_line_length, 
                                                     cv2.FONT_HERSHEY_PLAIN, 1.5, 1)[0]
            
            box_width = 500
            box_height = text_height * len(feedback_lines) + 20
            
            # Calculate box position
            box_x = 50
            box_y = y_position - box_height
            
            # Draw background box
            cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), 
                         self.colors['neutral'], -1)
            
            # Draw text lines
            for i, line in enumerate(feedback_lines):
                text_y = box_y + 25 + (i * 20)
                cv2.putText(frame, line, (box_x + 20, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)


# Legacy function for backward compatibility
def find_angle_and_display(img: np.ndarray, p1: int, p2: int, p3: int, 
                          keypoints: np.ndarray, confidence_threshold: float = 0.3,
                          draw: bool = True, correct_posture: bool = True, 
                          feedback: str = "") -> np.ndarray:
    """
    Legacy function for drawing angles and displaying feedback.
    
    Args:
        img (np.ndarray): Input frame
        p1, p2, p3 (int): Keypoint indices
        keypoints (np.ndarray): Keypoints array
        confidence_threshold (float): Minimum confidence
        draw (bool): Whether to draw
        correct_posture (bool): Posture correctness
        feedback (str): Feedback text
        
    Returns:
        np.ndarray: Frame with angle drawn
    """
    visualizer = Visualizer()
    return visualizer.draw_angle(img, p1, p2, p3, keypoints, confidence_threshold,
                                draw, correct_posture, feedback)


def display_bicep_count(frame: np.ndarray, bicep_count: int) -> np.ndarray:
    """
    Legacy function for displaying bicep count.
    
    Args:
        frame (np.ndarray): Input frame
        bicep_count (int): Bicep curl count
        
    Returns:
        np.ndarray: Frame with count displayed
    """
    visualizer = Visualizer()
    return visualizer.display_bicep_count(frame, bicep_count)


def display_squat_count(frame: np.ndarray, squat_count: int) -> np.ndarray:
    """
    Legacy function for displaying squat count.
    
    Args:
        frame (np.ndarray): Input frame
        squat_count (int): Squat count
        
    Returns:
        np.ndarray: Frame with count displayed
    """
    visualizer = Visualizer()
    return visualizer.display_squat_count(frame, squat_count) 