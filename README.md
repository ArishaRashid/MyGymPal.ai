# MyGymPal.ai - AI-Powered Fitness Application

**MyGymPal.ai** (formerly known as **Fitness Visionaire.ai**) is a cutting-edge fitness application that leverages artificial intelligence to provide real-time exercise form analysis, personalized feedback, and performance tracking. Built with MoveNet for pose detection and CustomANN for exercise classification, it offers an intelligent workout companion for fitness enthusiasts.

## 🎥 **Demo Video**

Watch MyGymPal.ai in action: [YouTube Demo](https://youtu.be/l61iYnCN4VI?si=JsGZXa8WdFL5yXij)

[![MyGymPal.ai Demo](https://img.youtube.com/vi/l61iYnCN4VI/0.jpg)](https://youtu.be/l61iYnCN4VI?si=JsGZXa8WdFL5yXij)

## 🔄 **How It Works**

MyGymPal.ai uses a sophisticated real-time processing pipeline to analyze exercise form and provide instant feedback:

```
Video Capture → Frame Processing → MoveNet Pose Detection → CustomANN Classification → Real-time Feedback
```

### **Technical Workflow**

1. **🎥 Video Capture**: Accesses the device's camera and captures video of the user performing exercises
2. **🖼️ Frame Processing**: Converts the video into individual frames for analysis
3. **🤖 MoveNet Detection**: Feeds each frame into the pre-trained MoveNet model
4. **📍 Keypoint Extraction**: MoveNet outputs 17 keypoints of the body for each frame
5. **🧠 AI Classification**: Passes the keypoints to the trained CustomANN classifier to evaluate exercise correctness
6. **✅ Form Analysis**: The classifier labels keypoints as correct/incorrect based on predefined criteria:
   - Angle of the elbow
   - Alignment of the wrist
   - Distance of hand from shoulder
   - Overall posture alignment
7. **🔄 Decision Loop**: 
   - **If Correct**: Continues processing the next frame
   - **If Incorrect**: Identifies faulty keypoints and associated body parts
8. **💬 User Feedback**: Provides specific instructions to correct posture:
   - "Keep your elbow close to your body"
   - "Do not bend your wrist"
   - "Lift your hand higher"
   - "Straighten your back"
9. **🔄 Continuous Monitoring**: Returns to frame processing to monitor corrections

This creates a real-time feedback loop that continuously analyzes and guides users toward proper exercise form.

## 🧠 **CustomANN Architecture**

MyGymPal.ai uses a custom Artificial Neural Network (CustomANN) for exercise form classification. Here's the detailed architecture:

### **Neural Network Structure**

#### **Dense Layers**
- **First Dense Layer**: 64 units with ReLU activation
  - Introduces non-linearity and mitigates vanishing gradient problem
  - Enables learning of complex patterns in pose data
- **Second Dense Layer**: 32 units with ReLU activation
  - Further feature extraction and pattern recognition
  - Maintains non-linearity for optimal learning

#### **Regularization Layers**
- **Dropout Layers**: 50% dropout rate
  - Prevents overfitting by randomly deactivating neurons during training
  - Improves model generalization on unseen data
  - Reduces reliance on specific neurons

#### **Normalization Layers**
- **Batch Normalization**: Applied after each dense layer
  - Stabilizes and speeds up training
  - Normalizes activations to mitigate internal covariate shifts
  - Ensures efficient network convergence

#### **Output Layer**
- **Single Unit**: Sigmoid activation function
  - Binary classification (correct/incorrect form)
  - Outputs probability between 0 and 1
  - Represents class prediction confidence

### **Training Configuration**

#### **Loss Function**
- **Binary Crossentropy**: Optimized for binary classification problems
- Handles probability-based predictions effectively

#### **Optimizer**
- **RMSprop**: Learning rate of 0.01
- Ideal for non-stationary objectives
- Adaptive learning rate across training process
- Efficient convergence on pose classification tasks

## 🚀 Features

### Core Functionality
- **Real-time Pose Detection**: Advanced pose tracking using MoveNet Lightning
- **Exercise Form Analysis**: AI-powered form evaluation and feedback
- **Rep Counting**: Automatic exercise repetition counting
- **Performance Tracking**: Real-time metrics and statistics
- **Visual Feedback**: On-screen annotations and guidance

### Supported Exercises
- **Bicep Curls**: Complete form analysis and rep counting
- **Squats**: Lower body exercise evaluation

### Technical Features
- **Low Latency**: Optimized for real-time processing
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Modular Architecture**: Easy to extend and customize
- **Comprehensive Logging**: Detailed performance monitoring

## 📁 Project Structure

```
MyGymPal.ai/
├── src/                          # Source code directory
│   ├── core/                     # Core functionality
│   │   ├── pose_detection.py     # MoveNet pose detection
│   │   └── visualization.py      # Visual rendering and UI
│   ├── ml/                       # Machine learning components
│   │   └── custom_classifier.py  # CustomANN model and analysis
│   └── app/                      # Application logic
│       └── main_app.py           # Main application entry point
├── config/                       # Configuration files
│   └── settings.py               # Application settings and constants
├── models/                       # Trained model files
│   ├── movenet_singlepose_lightning.tflite
│   ├── custom_ann_model.h5
│   └── label_encoders/
├── data/                         # Data files and datasets
├── tests/                        # Unit tests
├── docs/                         # Documentation
├── requirements.txt              # Python dependencies
└── README.md                    # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Webcam or video input device
- Sufficient RAM (4GB+ recommended)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/MyGymPal.ai.git
   cd MyGymPal.ai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download model files**
   ```bash
   # Ensure model files are in the models/ directory:
   # - models/movenet_singlepose_lightning.tflite
   # - models/custom_ann_model.h5
   # - models/label_encoder_bicepcurlphase.pkl
   # - models/label_encoder_orientation.pkl
   ```

## 🚀 Usage

### Quick Start

1. **Run the application**
   ```bash
   python src/app/main_app.py
   ```

2. **Position yourself in front of the camera**
   - Ensure good lighting
   - Stand at a reasonable distance (2-3 meters)
   - Wear form-fitting clothing for better detection

3. **Start exercising**
   - The application will automatically detect your pose
   - Real-time feedback will appear on screen
   - Rep counts and form analysis will be displayed

### Command Line Options

```bash
# Basic usage
python src/app/main_app.py

# With custom model paths
python src/app/main_app.py --movenet path/to/movenet.tflite --custom_ann path/to/custom_ann.h5

# Save output video
python src/app/main_app.py --output output_video.mp4
```

## 🔧 Configuration

### Model Paths
Edit `config/settings.py` to customize model file paths:

```python
MODEL_PATHS = {
    'movenet': 'path/to/movenet_model.tflite',
    'custom_ann': 'path/to/custom_ann_model.h5',
    'label_encoder_bicepcurl': 'path/to/bicep_encoder.pkl',
    'label_encoder_orientation': 'path/to/orientation_encoder.pkl'
}
```

### Exercise Parameters
Customize exercise analysis parameters:

```python
BICEP_CURL_CONFIG = {
    'open_angle_range': (140, 200),
    'closed_angle_range': (0, 40),
    'curl_up_threshold': 60,
    'curl_down_threshold': 120
}
```

### Visualization Settings
Adjust display parameters:

```python
COLORS = {
    'correct': (0, 255, 0),      # Green
    'incorrect': (0, 0, 255),    # Red
    'neutral': (255, 255, 255),  # White
    'count': (193, 111, 157)     # Purple
}
```

## 🧪 Development

### Project Structure Overview

#### Core Module (`src/core/`)
- **`pose_detection.py`**: MoveNet integration and pose analysis
- **`visualization.py`**: Drawing functions and UI components

#### Machine Learning Module (`src/ml/`)
- **`custom_classifier.py`**: CustomANN model for exercise classification

#### Application Module (`src/app/`)
- **`main_app.py`**: Main application logic and entry point

#### Configuration (`config/`)
- **`settings.py`**: All application settings and constants

### Adding New Exercises

1. **Define exercise parameters** in `config/settings.py`:
   ```python
   NEW_EXERCISE_CONFIG = {
       'name': 'New Exercise',
       'keypoints': ['shoulder', 'elbow', 'wrist'],
       'angles': ['shoulder_angle'],
       'phases': ['phase1', 'phase2']
   }
   ```

2. **Add analysis logic** in `src/ml/custom_classifier.py`:
   ```python
   def analyze_new_exercise(self, keypoints):
       # Add your analysis logic here
       pass
   ```

3. **Update visualization** in `src/core/visualization.py`:
   ```python
   def draw_new_exercise_angles(self, frame, keypoints):
       # Add visualization logic here
       pass
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_pose_detection.py
```

## 📊 Performance

### System Requirements
- **CPU**: Intel i5 or equivalent (4+ cores recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **GPU**: Optional, but recommended for better performance
- **Camera**: 720p minimum, 1080p recommended

### Performance Metrics
- **Latency**: <50ms per frame
- **FPS**: 30+ FPS on modern hardware
- **Accuracy**: >90% for supported exercises
- **Memory Usage**: <2GB RAM

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-feature`
3. **Make your changes** and add tests
4. **Run tests**: `pytest`
5. **Commit your changes**: `git commit -am 'Add new feature'`
6. **Push to the branch**: `git push origin feature/new-feature`
7. **Submit a pull request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## 🙏 Acknowledgments
- **MoveNet**: Google's real-time pose detection model
- **TensorFlow**: Machine learning framework
- **OpenCV**: Computer vision library
- **scikit-learn**: Machine learning utilities

## 📞 Support

- **Email**: arisharashidk@gmail.com
- **Issues**: [GitHub Issues](https://github.com/your-username/MyGymPal.ai/issues)
- **Documentation**: [Wiki](https://github.com/your-username/MyGymPal.ai/wiki)

## 🔄 Version History

- **v1.0.0**: Initial release with bicep curl and squat analysis

---

**MyGymPal.ai** - Empowering your fitness journey with AI! 💪🤖
