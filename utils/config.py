import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

MEDIAPIPE_CONFIDENCE = 0.7
MEDIAPIPE_TRACKING_CONFIDENCE = 0.5

GESTURE_CLASSES = ['idle', 'jump', 'left', 'right']
MODEL_CONFIDENCE_THRESHOLD = 0.6

JUMP_KEY = 'space'
LEFT_KEY = 'a'
RIGHT_KEY = 'd'

def ensure_directories():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)