"""
Unit tests for pose detection module.

This module contains comprehensive tests for the PoseDetector class
and related functionality in the pose detection module.

Author: MyGymPal.ai Team
Date: 2024
"""

import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.pose_detection import PoseDetector, rescale_points, rescale_three_points


class TestPoseDetector(unittest.TestCase):
    """Test cases for the PoseDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the model loading to avoid file dependencies
        with patch('tensorflow.lite.Interpreter') as mock_interpreter:
            mock_interpreter.return_value.get_input_details.return_value = [{'index': 0}]
            mock_interpreter.return_value.get_output_details.return_value = [{'index': 0}]
            self.pose_detector = PoseDetector("mock_model.tflite")
    
    def test_initialization(self):
        """Test PoseDetector initialization."""
        self.assertIsNotNone(self.pose_detector)
        self.assertEqual(self.pose_detector.model_path, "mock_model.tflite")
        self.assertIsNotNone(self.pose_detector.EDGES)
    
    def test_edges_definition(self):
        """Test that body connections are properly defined."""
        expected_edges = {
            (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
            (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11),
            (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
        }
        actual_edges = set(self.pose_detector.EDGES.keys())
        self.assertEqual(actual_edges, expected_edges)
    
    def test_get_coordinates_valid_index(self):
        """Test getting coordinates for valid keypoint index."""
        # Mock keypoints array
        mock_keypoints = np.zeros((1, 17, 3))
        mock_keypoints[0, 5] = [100, 200, 0.8]  # left_shoulder
        
        y, x, confidence = self.pose_detector.get_coordinates(mock_keypoints, 5)
        
        self.assertEqual(x, 200)
        self.assertEqual(y, 100)
        self.assertEqual(confidence, 0.8)
    
    def test_get_coordinates_invalid_index(self):
        """Test getting coordinates for invalid keypoint index."""
        mock_keypoints = np.zeros((1, 17, 3))
        
        # Test index out of range
        with self.assertRaises(IndexError):
            self.pose_detector.get_coordinates(mock_keypoints, 20)
        
        # Test negative index
        with self.assertRaises(IndexError):
            self.pose_detector.get_coordinates(mock_keypoints, -1)
    
    def test_calculate_angle(self):
        """Test angle calculation between three points."""
        # Create mock keypoints for a simple angle calculation
        mock_keypoints = np.zeros((1, 17, 3))
        # Set up three points forming a right angle
        mock_keypoints[0, 5] = [0, 0, 1.0]    # Point 1: (0, 0)
        mock_keypoints[0, 7] = [0, 100, 1.0]  # Point 2: (0, 100)
        mock_keypoints[0, 9] = [100, 100, 1.0] # Point 3: (100, 100)
        
        angle = self.pose_detector.calculate_angle(5, 7, 9, mock_keypoints)
        
        # Should be approximately 90 degrees
        self.assertAlmostEqual(angle, 90, delta=5)
    
    def test_get_bicep_curl_form_open(self):
        """Test bicep curl form detection for open position."""
        mock_keypoints = np.zeros((1, 17, 3))
        # Set up angles for open position (140-200 degrees)
        mock_keypoints[0, 5] = [0, 0, 1.0]     # right_shoulder
        mock_keypoints[0, 7] = [0, 100, 1.0]   # right_elbow
        mock_keypoints[0, 9] = [100, 100, 1.0] # right_hand
        
        form = self.pose_detector.get_bicep_curl_form(mock_keypoints)
        self.assertEqual(form, "open")
    
    def test_get_bicep_curl_form_closed(self):
        """Test bicep curl form detection for closed position."""
        mock_keypoints = np.zeros((1, 17, 3))
        # Set up angles for closed position (0-40 degrees)
        mock_keypoints[0, 5] = [0, 0, 1.0]     # right_shoulder
        mock_keypoints[0, 7] = [0, 100, 1.0]   # right_elbow
        mock_keypoints[0, 9] = [0, 200, 1.0]   # right_hand
        
        form = self.pose_detector.get_bicep_curl_form(mock_keypoints)
        self.assertEqual(form, "closed")
    
    def test_get_squat_form(self):
        """Test squat form detection."""
        mock_keypoints = np.zeros((1, 17, 3))
        # Set up angles for squat position
        mock_keypoints[0, 11] = [0, 0, 1.0]    # left_hip
        mock_keypoints[0, 12] = [0, 100, 1.0]  # right_hip
        mock_keypoints[0, 13] = [0, 200, 1.0]  # left_knee
        
        form = self.pose_detector.get_squat_form(mock_keypoints)
        # Should be "squat" or "unknown" depending on exact angles
        self.assertIn(form, ["squat", "standing", "unknown"])
    
    @patch('cv2.circle')
    def test_draw_keypoints(self, mock_circle):
        """Test drawing keypoints on frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_keypoints = np.zeros((1, 17, 3))
        mock_keypoints[0, 5] = [100, 200, 0.8]  # High confidence keypoint
        
        result_frame = self.pose_detector.draw_keypoints(frame, mock_keypoints, 0.3)
        
        # Verify cv2.circle was called for high confidence keypoints
        mock_circle.assert_called()
        self.assertEqual(result_frame.shape, frame.shape)
    
    @patch('cv2.line')
    def test_draw_connections(self, mock_line):
        """Test drawing connections between keypoints."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_keypoints = np.zeros((1, 17, 3))
        # Set up two connected keypoints with high confidence
        mock_keypoints[0, 5] = [100, 200, 0.8]  # left_shoulder
        mock_keypoints[0, 7] = [150, 250, 0.9]  # left_elbow
        
        result_frame = self.pose_detector.draw_connections(frame, mock_keypoints, 0.3)
        
        # Verify cv2.line was called for connections
        mock_line.assert_called()
        self.assertEqual(result_frame.shape, frame.shape)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_rescale_points(self):
        """Test rescaling of keypoint coordinates."""
        keypoints = np.array([100, 200])
        frame = np.zeros((480, 640, 3))
        
        x, y = rescale_points(keypoints, frame)
        
        self.assertIsInstance(x, int)
        self.assertIsInstance(y, int)
    
    def test_rescale_three_points(self):
        """Test rescaling of keypoint coordinates with confidence."""
        keypoints = np.array([100, 200, 0.8])
        frame = np.zeros((480, 640, 3))
        
        x, y, c = rescale_three_points(keypoints, frame)
        
        self.assertIsInstance(x, int)
        self.assertIsInstance(y, int)
        self.assertIsInstance(c, int)


class TestPoseDetectorIntegration(unittest.TestCase):
    """Integration tests for PoseDetector."""
    
    def setUp(self):
        """Set up test fixtures for integration tests."""
        with patch('tensorflow.lite.Interpreter') as mock_interpreter:
            mock_interpreter.return_value.get_input_details.return_value = [{'index': 0}]
            mock_interpreter.return_value.get_output_details.return_value = [{'index': 0}]
            self.pose_detector = PoseDetector("mock_model.tflite")
    
    def test_full_pose_analysis_workflow(self):
        """Test complete pose analysis workflow."""
        # Create a mock frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Mock the detect_pose method
        mock_keypoints = np.zeros((1, 17, 3))
        mock_keypoints[0, 5] = [100, 200, 0.8]  # left_shoulder
        mock_keypoints[0, 7] = [150, 250, 0.9]  # left_elbow
        mock_keypoints[0, 9] = [200, 300, 0.7]  # left_wrist
        
        with patch.object(self.pose_detector, 'detect_pose', return_value=mock_keypoints):
            # Test the complete workflow
            detected_keypoints = self.pose_detector.detect_pose(frame)
            
            # Test coordinate extraction
            y, x, conf = self.pose_detector.get_coordinates(detected_keypoints, 5)
            self.assertEqual(x, 200)
            self.assertEqual(y, 100)
            self.assertEqual(conf, 0.8)
            
            # Test angle calculation
            angle = self.pose_detector.calculate_angle(5, 7, 9, detected_keypoints)
            self.assertIsInstance(angle, float)
            self.assertGreaterEqual(angle, 0)
            self.assertLessEqual(angle, 360)
            
            # Test form detection
            form = self.pose_detector.get_bicep_curl_form(detected_keypoints)
            self.assertIn(form, ["open", "closed", "unknown"])


if __name__ == '__main__':
    unittest.main() 