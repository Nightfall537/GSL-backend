#!/usr/bin/env python3
"""
Complete GSL LSTM Training Pipeline
Trains fusion model with MediaPipe landmarks + ResNeXt embeddings
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import numpy as np
from pathlib import Path
import tensorflow as tf
from datetime import datetime

from gsl_training_pipeline.utils.logger import setup_logger
from gsl_training_pipeline.data_generators.sequence_generator import create_data_generators
from gsl_training_pipeline.models.fusion_lstm import build_fusion_model
from gsl_training_pipeline.config.training_config import (
    OUTPUT_DIR, MODELS_DIR, LOGS_DIR, TRAINING_CONFIG, MODEL_CONFIG
)

logger = setup_logger(__name__, log_file='logs/training.log')


def load_dataset():
    """Load complete dataset"""
    logger.info("📂 Loading dataset...")
    
    # Load landmarks
    landmarks_file = OUTPUT_DIR / "gsl_landmarks.json"
    with open(landmarks_file, 'r', encoding='utf-8') as f:
        landmarks_data = json.load(f)
    
    # Load embeddings
    embeddings_file = OUTPUT_DIR / "resnext_embeddings.npy"
    embeddings = np.load(embeddings_file)
    
    # Load class index
    class_index_file = OUTPUT_DIR / "class_index.json"
    with open(class_index_file, 'r', encoding='utf-8') as f:
        class_index_data = json.load(f)
    
    # Extract class_to_idx (remove 'num_classes' key)
    class_to_idx = {k: v for k, v in class_index_data.items() if k != 'num_classes'}
    
    # Extract data
    clips = landmarks_data['clips']
    landmarks = [np.array(c['landmarks'], dtype=np.float32) for c in clips]
    labels = [c['label'] for c in clips]
    
    logger.info(f"✅ Loaded {len(landmarks)} clips")
    logger.info(f"✅ {len(class_to_idx)} classes")
    logger.info(f"✅ Landmarks shape: variable × 468")
    logger.info(f"✅ Embeddings shape: {embeddings.shape}")
    
    return landmarks, embeddings, labels, class_to_idx


def create_callbacks(model_name: str):
    """Create training callbacks"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = []
    
    # Early Stopping
    if TRAINING_CONFIG['early_stopping']['enabled']:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor=TRAINING_CONFIG['early_stopping']['monitor'],
            patience=TRAINING_CONFIG['early_stopping']['patience'],
            restore_best_weights=TRAINING_CONFIG['early_stopping']['restore_best_weights'],
            min_delta=TRAINING_CONFIG['early_stopping']['min_delta'],
            verbose=1
        )
        callbacks.append(early_stop)
        logger.info("✅ Early stopping enabled")
    
    # Model Checkpoint
    if TRAINING_CONFIG['model_checkpoint']['enabled']:
        checkpoint_path = MODELS_DIR / f"{model_name}_{timestamp}_best.h5"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=TRAINING_CONFIG['model_checkpoint']['monitor'],
            save_best_only=TRAINING_CONFIG['model_checkpoint']['save_best_only'],
            mode=TRAINING_CONFIG['model_checkpoint']['mode'],
            verbose=1
        )
        callbacks.append(checkpoint)
        logger.info(f"✅ Model checkpoint: {checkpoint_path}")
    
    # Reduce LR on Plateau
    if TRAINING_CONFIG['reduce_lr']['enabled']:
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor=TRAINING_CONFIG['reduce_lr']['monitor'],
            factor=TRAINING_CONFIG['reduce_lr']['factor'],
            patience=TRAINING_CONFIG['reduce_lr']['patience'],
            min_lr=TRAINING_CONFIG['reduce_lr']['min_lr'],
            verbose=TRAINING_CONFIG['reduce_lr']['verbose']
        )
        callbacks.append(reduce_lr)
        logger.info("✅ Learning rate scheduler enabled")
    
    # TensorBoard
    if TRAINING_CONFIG['tensorboard']['enabled']:
        tensorboard_dir = Path(TRAINING_CONFIG['tensorboard']['log_dir']) / timestamp
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=TRAINING_CONFIG['tensorboard']['histogram_freq'],
            write_graph=TRAINING_CONFIG['tensorboard']['write_graph']
        )
        callbacks.append(tensorboard)
        logger.info(f"✅ TensorBoard: {tensorboard_dir}")
    
    return callbacks


def train_model():
    """Complete training pipeline"""
    logger.info("=" * 80)
    logger.info("🚀 GSL LSTM Training Pipeline")
    logger.info("=" * 80)
    logger.info(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load dataset
    landmarks, embeddings, labels, class_to_idx = load_dataset()
    num_classes = len(class_to_idx)
    
    # Create data generators
    logger.info("\n📊 Creating data generators...")
    train_gen, val_gen, _ = create_data_generators(
        landmarks=landmarks,
        embeddings=embeddings,
        labels=labels,
        class_to_idx=class_to_idx,
        train_split=TRAINING_CONFIG['train_split'],
        batch_size=TRAINING_CONFIG['batch_size'],
        random_seed=TRAINING_CONFIG['random_seed']
    )
    
    # Build model
    logger.info("\n🏗️  Building fusion model...")
    model_config = {
        **MODEL_CONFIG,
        'learning_rate': TRAINING_CONFIG['learning_rate']
    }
    model = build_fusion_model(
        num_classes=num_classes,
        config=model_config
    )
    
    # Create callbacks
    logger.info("\n⚙️  Setting up callbacks...")
    callbacks = create_callbacks('gsl_fusion_lstm')
    
    # Train model
    logger.info("\n" + "=" * 80)
    logger.info("🎯 Starting Training")
    logger.info("=" * 80)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=TRAINING_CONFIG['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = MODELS_DIR / f"gsl_fusion_lstm_final_{timestamp}.h5"
    model.save(str(final_model_path))
    logger.info(f"\n💾 Saved final model: {final_model_path}")
    
    # Save training history
    history_path = MODELS_DIR / f"training_history_{timestamp}.json"
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    logger.info(f"💾 Saved training history: {history_path}")
    
    # Print final metrics
    logger.info("\n" + "=" * 80)
    logger.info("✅ Training Complete!")
    logger.info("=" * 80)
    logger.info(f"📊 Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    logger.info(f"📊 Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    logger.info(f"📊 Final Training Loss: {history.history['loss'][-1]:.4f}")
    logger.info(f"📊 Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    
    if 'top_3_accuracy' in history.history:
        logger.info(f"📊 Final Top-3 Accuracy: {history.history['val_top_3_accuracy'][-1]:.4f}")
    
    logger.info("=" * 80)
    logger.info(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return model, history


def main():
    """Main training function"""
    try:
        # Enable GPU memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"🎮 GPU available: {len(gpus)} device(s)")
        else:
            logger.info("💻 Running on CPU")
        
        # Train model
        model, history = train_model()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
