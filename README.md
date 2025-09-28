# Jump King Hand Gesture Controller

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green?style=flat-square&logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange?style=flat-square&logo=google)
![YOLO](https://img.shields.io/badge/YOLO-v8-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

Real-time hand gesture recognition system for hands-free gaming control.

## Overview

A computer vision system that enables hands-free control of Jump King through real-time hand gesture recognition. Built with Python, MediaPipe, OpenCV, and YOLO for robust hand tracking, combined with machine learning for accurate gesture classification.

## Technologies Used

- **Python** - Main development language
- **OpenCV** - Computer vision and video processing
- **MediaPipe** - Hand landmark detection (21 points)
- **YOLO v8** - Enhanced object detection
- **scikit-learn** - Random Forest classification
- **pynput** - Keyboard input simulation
- **NumPy** - Numerical computing

## Features

- Real-time hand tracking at 30+ FPS
- Machine learning gesture classification with Random Forest
- Multi-modal detection combining MediaPipe and YOLO
- Gesture smoothing with temporal filtering
- Low-latency control optimized for gaming
- Fallback modes when MediaPipe unavailable

## Supported Gestures

| Gesture | Action | Key |
|---------|--------|-----|
| Closed Fist | Jump | Space |
| Open Palm Left | Move Left | A |
| Open Palm Right | Move Right | D |
| Open Palm Up | Idle | None |

---

## Demo

### Video Processing Demo
```bash
python video_demo.py
```
*Creates and processes demo video showing gesture detection pipeline*

### Live Camera Demo (if available)
```bash
python simple_gesture.py
```
*Real-time gesture detection with camera feed*

### Component Testing
```bash
python comprehensive_test.py
```

## Quick Start

### Video Demo (No Camera Required)
```bash
pip install opencv-python numpy
python video_demo.py
```

### Live Camera Demo  
```bash
pip install opencv-python mediapipe
python simple_gesture.py
```

### Full System
```bash
pip install -r requirements.txt
python src/main.py
```

### System Testing
```bash
python comprehensive_test.py
```

## Installation

### Prerequisites
- Python 3.8+
- Webcam (optional)

### Setup
```bash
git clone https://github.com/IliasSoultana/Jump-King-Hand-Gesture-Controller.git
cd Jump-King-Hand-Gesture-Controller
pip install -r requirements.txt
python comprehensive_test.py
```

### Manual Installation
```bash
pip install opencv-python scikit-learn numpy pynput joblib
pip install ultralytics mediapipe  # Optional enhancements
```

## Project Structure

```
Jump-King-Hand-Gesture-Controller/
├── src/                     # Core application
│   ├── main.py              # Main application entry
│   ├── hand_detector.py     # MediaPipe hand tracking
│   ├── yolo_detector.py     # YOLO object detection  
│   ├── gesture_classifier.py  # ML gesture classification
│   └── game_controller.py  # Keyboard input control
├── models/                  # Trained ML models
├── data/                    # Training datasets
├── utils/                   # Configuration files
├── video_demo.py            # Video processing demo
├── camera_test.py           # Camera detection utility
├── comprehensive_test.py    # Full system testing
└── README.md                # Documentation
```

---

## Machine Learning Pipeline

```
Hand Detection → Landmark Extraction → Feature Engineering → ML Classification → Gesture Smoothing → Keyboard Control
```

### Pipeline Components
- Hand Detection: 21 3D landmarks extracted
- Feature Engineering: 13 features (5 distances + 8 angles) 
- ML Classification: Random Forest classifier
- Gesture Smoothing: 5-frame temporal filter

## Technical Implementation

### Feature Extraction
- 21 hand landmarks (x, y, z coordinates)
- 5 distance measurements (fingertips to wrist)  
- 8 angle calculations (finger orientations)
- StandardScaler normalization

### Machine Learning
- Random Forest Classifier (100 estimators)
- Max depth: 10 for optimal generalization
- 5-fold cross-validation during training
- Confidence threshold: 0.6 for prediction acceptance

### Performance
- Real-time processing: 30+ FPS
- Gesture-to-action latency: <50ms
- Temporal smoothing: 5-frame window
- Memory usage: <200MB

## Usage

### Basic Usage
1. Run: `python src/main.py`
2. Position hand in front of camera
3. Make gestures:
   - Fist = Jump (Space)
   - Palm Left = Move Left (A)
   - Palm Right = Move Right (D)
   - Open Palm = Idle

### Camera-Free Testing
```bash
python video_demo.py      # Process demo video
python opencv_demo.py     # OpenCV-only detection
```

### Configuration
Edit `utils/config.py` to customize:
- Gesture sensitivity thresholds
- Key mappings
- Detection confidence levels
- Smoothing window size

### Multi-Modal Detection
- **Primary**: MediaPipe hand landmark detection
- **Secondary**: YOLO v8 object detection for robustness
- **Fallback**: OpenCV-only contour detection

### Gesture Smoothing
- Temporal filtering with 5-frame sliding window
- Confidence-based prediction weighting
- Adaptive thresholding for different lighting conditions

### Real-time Optimization
- Efficient feature extraction pipeline
- Lazy loading of ML models
- Memory-optimized video processing
- Multi-threaded gesture classification

---

## Installation

### Prerequisites
- Python 3.8+ (Python 3.13+ recommended)
- Webcam (optional - demos work without camera)
- Windows/Linux/macOS

### Quick Setup
```bash
git clone https://github.com/IliasSoultana/Jump-King-Hand-Gesture-Controller.git
cd Jump-King-Hand-Gesture-Controller

pip install -r requirements.txt

python comprehensive_test.py
```

### Manual Installation
```bash
# Core dependencies
pip install opencv-python scikit-learn numpy pynput joblib

# Optional enhanced detection
pip install ultralytics mediapipe  # MediaPipe requires Python <3.13
```

---

## Usage

### Quick Start
1. Run: `python src/main.py` (or `python video_demo.py` for camera-free demo)
2. Position hand in front of camera
3. Control with gestures:
   - **Fist** = Jump (Space)
   - **Palm Left** = Move Left (A)  
   - **Palm Right** = Move Right (D)
   - **Open Palm** = Idle

### Camera-Free Testing
```bash
python video_demo.py           # Process demo video
python opencv_demo.py          # OpenCV-only detection
python comprehensive_test.py   # Full system validation
```

---

## Development

### Adding New Gestures
1. Define gesture in `gesture_classifier.py`
2. Add training data generation
3. Update key mappings in `game_controller.py`
4. Retrain: `python -c "from src.gesture_classifier import GestureClassifier; GestureClassifier().train_model()"`

### Testing Components
```bash
python comprehensive_test.py  # Full system test
python camera_test.py         # Camera detection
python demo_test.py          # Component validation
```

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

### Contributing

Contributions welcome! Process:
1. Fork repository
2. Create feature branch
3. Run tests: `python comprehensive_test.py`
4. Submit pull request

Focus areas: gesture accuracy, new game support, performance optimization

---

## Acknowledgments

- MediaPipe - Hand tracking framework
- Ultralytics - YOLO implementation  
- OpenCV - Computer vision library
- scikit-learn - Machine learning toolkit