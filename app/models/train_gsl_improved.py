#!/usr/bin/env python3
"""
Improved GSL Training with Better Fusion and Data Handling
Optimized for small dataset with many classes
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import json
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from datetime import datetime
from collections import Counter

from gsl_training_pipeline.utils.logger import setup_logger
from gsl_training_pipeline.config.training_config import OUTPUT_DIR, MODELS_DIR

logger = setup_logger(__name__, log_file='logs/training_improved.log')


def load_dataset():
    """Load dataset"""
    logger.info("📂 Loading dataset...")
    
    landmarks_file = OUTPUT_DIR / "gsl_landmarks.json"
    with open(landmarks_file, 'r', encoding='utf-8') as f:
        landmarks_data = json.load(f)
    
    embeddings_file = OUTPUT_DIR / "resnext_embeddings.npy"
    embeddings = np.load(embeddings_file)
    
    class_index_file = OUTPUT_DIR / "class_index.json"
    with open(class_index_file, 'r', encoding='utf-8') as f:
        class_index_data = json.load(f)
    
    class_to_idx = {k: v for k, v in class_index_data.items() if k != 'num_classes'}
    
    clips = landmarks_data['clips']
    landmarks = [np.array(c['landmarks'], dtype=np.float32) for c in clips]
    labels = [c['label'] for c in clips]
    
    logger.info(f"✅ Loaded {len(landmarks)} clips, {len(class_to_idx)} classes")
    
    return landmarks, embeddings, labels, class_to_idx


def build_improved_fusion_model(num_classes: int) -> Model:
    """
    Improved fusion model with attention and better feature integration
    """
    logger.info("🏗️  Building Improved Fusion Model...")
    
    # Landmark input with masking
    landmark_input = layers.Input(shape=(None, 468), name='landmark_input')
    mask_input = layers.Input(shape=(None,), dtype='bool', name='mask_input')
    
    # Bidirectional LSTM with attention
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.3)
    )(landmark_input, mask=mask_input)
    x = layers.BatchNormalization()(x)
    
    x = layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.3)
    )(x)
    x = layers.BatchNormalization()(x)
    
    # Attention mechanism
    attention = layers.Dense(1, activation='tanh')(x)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(256)(attention)
    attention = layers.Permute([2, 1])(attention)
    
    # Apply attention
    x_attended = layers.Multiply()([x, attention])
    landmark_features = layers.GlobalAveragePooling1D()(x_attended)
    landmark_features = layers.Dropout(0.4)(landmark_features)
    
    # ResNeXt branch with more capacity
    resnext_input = layers.Input(shape=(512,), name='resnext_input')
    y = layers.Dense(256, activation='relu')(resnext_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.3)(y)
    y = layers.Dense(128, activation='relu')(y)
    y = layers.BatchNormalization()(y)
    resnext_features = layers.Dropout(0.3)(y)
    
    # Fusion with gating mechanism
    fused = layers.Concatenate()([landmark_features, resnext_features])
    fused = layers.BatchNormalization()(fused)
    
    # Classification head
    z = layers.Dense(256, activation='relu')(fused)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.5)(z)
    
    z = layers.Dense(128, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.4)(z)
    
    output = layers.Dense(num_classes, activation='softmax')(z)
    
    model = Model(
        inputs=[landmark_input, resnext_input, mask_input],
        outputs=output,
        name='GSL_Improved_Fusion'
    )
    
    # Use lower learning rate for small dataset
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_acc')]
    )
    
    logger.info(f"📊 Model parameters: {model.count_params():,}")
    
    return model


class ImprovedDataGenerator(keras.utils.Sequence):
    """Data generator with strong augmentation"""
    
    def __init__(self, landmarks, embeddings, labels, batch_size=8, shuffle=True, augment=False):
        self.landmarks = landmarks
        self.embeddings = embeddings
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(landmarks))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.landmarks) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_landmarks = [self.landmarks[i] for i in batch_indices]
        batch_embeddings = self.embeddings[batch_indices]
        batch_labels = self.labels[batch_indices]
        
        max_len = max(len(seq) for seq in batch_landmarks)
        
        padded_landmarks = np.zeros((len(batch_landmarks), max_len, 468), dtype=np.float32)
        masks = np.zeros((len(batch_landmarks), max_len), dtype=np.bool_)
        
        for i, seq in enumerate(batch_landmarks):
            seq_len = len(seq)
            
            if self.augment:
                # Strong augmentation
                seq = self._augment(seq)
            
            padded_landmarks[i, :seq_len] = seq
            masks[i, :seq_len] = True
        
        inputs = {
            'landmark_input': padded_landmarks,
            'resnext_input': batch_embeddings,
            'mask_input': masks
        }
        
        return inputs, batch_labels
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _augment(self, sequence):
        """Strong data augmentation"""
        seq = sequence.copy()
        
        # 1. Gaussian noise
        noise = np.random.normal(0, 0.02, seq.shape)
        seq = seq + noise
        
        # 2. Random scaling (simulate distance variation)
        scale = np.random.uniform(0.9, 1.1)
        seq = seq * scale
        
        # 3. Random time warping (slight speed variation)
        if len(seq) > 10 and np.random.rand() > 0.5:
            indices = np.sort(np.random.choice(len(seq), size=len(seq), replace=True))
            seq = seq[indices]
        
        # 4. Random frame dropout (simulate occlusion)
        if np.random.rand() > 0.7:
            dropout_mask = np.random.rand(len(seq)) > 0.1
            seq = seq * dropout_mask[:, np.newaxis]
        
        seq = np.clip(seq, 0, 1)
        return seq.astype(np.float32)


def train_improved_model():
    """Train with improved approach"""
    logger.info("=" * 80)
    logger.info("🚀 Improved GSL Training Pipeline")
    logger.info("=" * 80)
    
    # Load data
    landmarks, embeddings, labels, class_to_idx = load_dataset()
    num_classes = len(class_to_idx)
    
    # Convert labels
    label_indices = np.array([class_to_idx[label] for label in labels])
    labels_onehot = keras.utils.to_categorical(label_indices, num_classes)
    
    # Analyze class distribution
    label_counts = Counter(labels)
    logger.info(f"\n📊 Class distribution:")
    logger.info(f"   Classes with 1 sample: {sum(1 for c in label_counts.values() if c == 1)}")
    logger.info(f"   Classes with 2+ samples: {sum(1 for c in label_counts.values() if c >= 2)}")
    logger.info(f"   Max samples per class: {max(label_counts.values())}")
    
    # Split data - use more for training given small dataset
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(landmarks))
    train_idx, val_idx = train_test_split(indices, train_size=0.8, random_state=42)
    
    train_landmarks = [landmarks[i] for i in train_idx]
    train_embeddings = embeddings[train_idx]
    train_labels = labels_onehot[train_idx]
    
    val_landmarks = [landmarks[i] for i in val_idx]
    val_embeddings = embeddings[val_idx]
    val_labels = labels_onehot[val_idx]
    
    logger.info(f"\n📊 Data split: {len(train_landmarks)} train, {len(val_landmarks)} val")
    
    # Create generators with smaller batch size
    train_gen = ImprovedDataGenerator(
        train_landmarks, train_embeddings, train_labels,
        batch_size=8, shuffle=True, augment=True
    )
    
    val_gen = ImprovedDataGenerator(
        val_landmarks, val_embeddings, val_labels,
        batch_size=8, shuffle=False, augment=False
    )
    
    # Build model
    model = build_improved_fusion_model(num_classes)
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=30,  # More patience for small dataset
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / f"gsl_improved_{timestamp}_best.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train
    logger.info("\n🎯 Starting Training...")
    logger.info("=" * 80)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=200,  # More epochs with early stopping
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_path = MODELS_DIR / f"gsl_improved_final_{timestamp}.h5"
    model.save(str(final_path))
    
    # Results
    logger.info("\n" + "=" * 80)
    logger.info("✅ Training Complete!")
    logger.info("=" * 80)
    logger.info(f"📊 Best Val Accuracy: {max(history.history['val_accuracy']):.4f}")
    logger.info(f"📊 Best Val Top-3: {max(history.history['val_top_3_acc']):.4f}")
    logger.info(f"📊 Final Train Accuracy: {history.history['accuracy'][-1]:.4f}")
    logger.info(f"📊 Final Val Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    logger.info("=" * 80)
    
    return model, history


if __name__ == "__main__":
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        model, history = train_improved_model()
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        sys.exit(1)
