"""
Configuration settings for Sign Language Recognition System
"""

import os

# Application Settings
APP_NAME = "Sign Language Recognition System"
VERSION = "1.0.0"

# Camera Settings
DEFAULT_CAMERA_ID = 0
TARGET_FPS = 30
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# MediaPipe Settings
MEDIAPIPE_CONFIDENCE = 0.7
MEDIAPIPE_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 2

# AI Model Settings
SEQUENCE_LENGTH = 30  # Number of frames for gesture recognition
MODEL_INPUT_SIZE = 1890  # 30 frames × 21 landmarks × 3 coordinates
CONFIDENCE_THRESHOLD = 0.7
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DENSE_UNITS = 32
DROPOUT_RATE = 0.2

# Training Settings
BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001

# File Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
TRAINING_DIR = os.path.join(DATA_DIR, "training")
VOCABULARIES_DIR = os.path.join(DATA_DIR, "vocabularies")

# Performance Settings
MAX_PROCESSING_TIME_MS = 500  # Maximum processing time per frame
HISTORY_SIZE = 10  # Number of recent gestures to keep in history

# UI Settings
WINDOW_TITLE = f"{APP_NAME} v{VERSION}"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

# Text-to-Speech Settings
TTS_RATE = 150  # Words per minute
TTS_VOLUME = 0.9

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"