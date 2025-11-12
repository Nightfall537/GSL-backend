"""
Training Configuration for GSL LSTM Model
Defines all hyperparameters and paths for the training pipeline
"""
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent.parent.parent
SEGMENTED_CLIPS_DIR = BASE_DIR / "segmented_clips"
METADATA_FILE = SEGMENTED_CLIPS_DIR / "segmentation_metadata.json"
OUTPUT_DIR = BASE_DIR / "gsl_dataset"
MODELS_DIR = BASE_DIR / "trained_models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================
FEATURE_CONFIG = {
    # MediaPipe Holistic settings
    'mediapipe': {
        'model_complexity': 2,  # 0, 1, or 2 (highest accuracy)
        'min_detection_confidence': 0.5,
        'min_tracking_confidence': 0.5,
        'enable_segmentation': True,
        'refine_face_landmarks': True,
        'static_image_mode': False
    },
    
    # ResNeXt 3D CNN settings
    'resnext': {
        'model_name': 'r3d_18',  # or 'mc3_18'
        'pretrained': True,
        'num_frames': 16,  # Sample 16 frames uniformly
        'frame_size': (112, 112),  # Resize to 112x112
        'normalize_mean': [0.485, 0.456, 0.406],  # ImageNet stats
        'normalize_std': [0.229, 0.224, 0.225],
        'embedding_dim': 512
    },
    
    # Feature dimensions
    'landmark_dim': 468,  # Fixed: pose(132) + hands(126) + face(210)
    'embedding_dim': 512  # ResNeXt output dimension
}

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
MODEL_CONFIG = {
    'model_type': 'fusion',  # 'landmarks_only' or 'fusion'
    
    # LSTM settings
    'lstm_units': 256,
    'lstm_layers': 2,
    'bidirectional': True,
    'dropout_rate': 0.4,
    'recurrent_dropout': 0.2,
    
    # Dense layer settings
    'dense_units': [512, 256],  # Hidden layer sizes
    'dense_dropout': [0.4, 0.3],  # Dropout after each dense layer
    
    # Fusion settings (only for fusion model)
    'fusion_dense_units': [768, 512, 256],
    'fusion_dropout': [0.4, 0.3],
    
    # Output
    'activation': 'relu',
    'output_activation': 'softmax'
}

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================
TRAINING_CONFIG = {
    # Data split
    'train_split': 0.7,
    'val_split': 0.3,
    'stratify': True,
    'random_seed': 42,
    
    # Training
    'batch_size': 16,
    'epochs': 100,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    'loss': 'categorical_crossentropy',
    'metrics': ['accuracy'],
    
    # Regularization
    'l2_weight_decay': 0.0005,
    'use_batch_norm': True,
    
    # Data augmentation
    'augmentation': {
        'enabled': True,
        'gaussian_noise_std': 0.01,
        'apply_to_training_only': True
    },
    
    # Callbacks
    'early_stopping': {
        'enabled': True,
        'monitor': 'val_loss',
        'patience': 15,
        'restore_best_weights': True,
        'min_delta': 0.001
    },
    
    'model_checkpoint': {
        'enabled': True,
        'monitor': 'val_accuracy',
        'save_best_only': True,
        'mode': 'max',
        'save_weights_only': False
    },
    
    'reduce_lr': {
        'enabled': True,
        'monitor': 'val_loss',
        'factor': 0.5,
        'patience': 5,
        'min_lr': 1e-6,
        'verbose': 1
    },
    
    'tensorboard': {
        'enabled': True,
        'log_dir': str(LOGS_DIR / 'tensorboard'),
        'histogram_freq': 1,
        'write_graph': True
    },
    
    # K-Fold Cross Validation
    'kfold': {
        'enabled': False,  # Set to True for K-fold CV
        'n_splits': 5,
        'shuffle': True,
        'random_state': 42
    }
}

# ============================================================================
# INFERENCE SETTINGS
# ============================================================================
INFERENCE_CONFIG = {
    'latency_target_ms': 80,  # Target inference time
    'confidence_threshold': 0.7,  # Minimum confidence for prediction
    'buffer_size': 30,  # Number of frames to buffer for gesture detection
    'motion_threshold': 0.05,  # Threshold for detecting gesture start/end
    'display_fps': True,
    'display_landmarks': True
}

# ============================================================================
# LOGGING
# ============================================================================
LOGGING_CONFIG = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': str(LOGS_DIR / 'training.log'),
    'console_output': True
}

# ============================================================================
# HARDWARE
# ============================================================================
HARDWARE_CONFIG = {
    'use_gpu': True,  # Automatically detect and use GPU if available
    'gpu_memory_growth': True,  # Enable memory growth for TensorFlow
    'mixed_precision': False,  # Use FP16 for faster training (requires GPU)
    'num_workers': 4,  # Number of workers for data loading
    'prefetch_buffer': 2  # Number of batches to prefetch
}

# ============================================================================
# VALIDATION
# ============================================================================
VALIDATION_CONFIG = {
    'min_clips_per_class': 1,  # Minimum clips required per class
    'max_sequence_length': 200,  # Maximum frames per sequence
    'min_sequence_length': 10,  # Minimum frames per sequence
    'validate_dimensions': True,
    'check_nan_values': True
}
