#!/usr/bin/env python3
"""
Final Optimized GSL Training
Focus on classes with sufficient samples for reliable training
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

logger = setup_logger(__name__, log_file='logs/training_final.log')


def load_and_filter_dataset(min_samples=2):
    """Load dataset and filter to classes with sufficient samples"""
    logger.info("📂 Loading and filtering dataset...")
    
    landmarks_file = OUTPUT_DIR / "gsl_landmarks.json"
    with open(landmarks_file, 'r', encoding='utf-8') as f:
        landmarks_data = json.load(f)
    
    embeddings_file = OUTPUT_DIR / "resnext_embeddings.npy"
    all_embeddings = np.load(embeddings_file)
    
    clips = landmarks_data['clips']
    all_landmarks = [np.array(c['landmarks'], dtype=np.float32) for c in clips]
    all_labels = [c['label'] for c in clips]
    
    # Count samples per class
    label_counts = Counter(all_labels)
    valid_classes = {label for label, count in label_counts.items() if count >= min_samples}
    
    logger.info(f"📊 Total classes: {len(label_counts)}")
    logger.info(f"📊 Classes with {min_samples}+ samples: {len(valid_classes)}")
    
    # Filter to valid classes
    filtered_landmarks = []
    filtered_embeddings = []
    filtered_labels = []
    
    for i, label in enumerate(all_labels):
        if label in valid_classes:
            filtered_landmarks.append(all_landmarks[i])
            filtered_embeddings.append(all_embeddings[i])
            filtered_labels.append(label)
    
    # Create new class mapping
    unique_labels = sorted(valid_classes)
    class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    logger.info(f"✅ Filtered to {len(filtered_landmarks)} clips, {len(class_to_idx)} classes")
    logger.info(f"📋 Classes: {', '.join(unique_labels)}")
    
    return filtered_landmarks, np.array(filtered_embeddings), filtered_labels, class_to_idx


def build_optimized_fusion_model(num_classes: int) -> Model:
    """Optimized fusion model"""
    logger.info("🏗️  Building Optimized Fusion Model...")
    
    # Inputs
    landmark_input = layers.Input(shape=(None, 468), name='landmark_input')
    resnext_input = layers.Input(shape=(512,), name='resnext_input')
    mask_input = layers.Input(shape=(None,), dtype='bool', name='mask_input')
    
    # Landmark branch - simpler for small dataset
    x = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, dropout=0.3)
    )(landmark_input, mask=mask_input)
    x = layers.BatchNormalization()(x)
    landmark_features = layers.Dropout(0.4)(x)
    
    # ResNeXt branch
    y = layers.Dense(128, activation='relu')(resnext_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.3)(y)
    resnext_features = y
    
    # Fusion
    fused = layers.Concatenate()([landmark_features, resnext_features])
    fused = layers.BatchNormalization()(fused)
    
    # Classification
    z = layers.Dense(128, activation='relu')(fused)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.5)(z)
    
    z = layers.Dense(64, activation='relu')(z)
    z = layers.Dropout(0.4)(z)
    
    output = layers.Dense(num_classes, activation='softmax')(z)
    
    model = Model(
        inputs=[landmark_input, resnext_input, mask_input],
        outputs=output,
        name='GSL_Optimized_Fusion'
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"📊 Model parameters: {model.count_params():,}")
    
    return model


class OptimizedDataGenerator(keras.utils.Sequence):
    """Optimized data generator"""
    
    def __init__(self, landmarks, embeddings, labels, batch_size=4, shuffle=True, augment=False):
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
        """Data augmentation"""
        seq = sequence.copy()
        
        # Gaussian noise
        noise = np.random.normal(0, 0.015, seq.shape)
        seq = seq + noise
        
        # Random scaling
        scale = np.random.uniform(0.95, 1.05)
        seq = seq * scale
        
        seq = np.clip(seq, 0, 1)
        return seq.astype(np.float32)


def train_final_model():
    """Final optimized training"""
    logger.info("=" * 80)
    logger.info("🚀 Final Optimized GSL Training")
    logger.info("=" * 80)
    
    # Load filtered dataset
    landmarks, embeddings, labels, class_to_idx = load_and_filter_dataset(min_samples=2)
    num_classes = len(class_to_idx)
    
    # Convert labels
    label_indices = np.array([class_to_idx[label] for label in labels])
    labels_onehot = keras.utils.to_categorical(label_indices, num_classes)
    
    # Split data
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(landmarks))
    
    # Try stratified split
    try:
        train_idx, val_idx = train_test_split(
            indices, train_size=0.75, stratify=label_indices, random_state=42
        )
        logger.info("✅ Using stratified split")
    except:
        train_idx, val_idx = train_test_split(
            indices, train_size=0.75, random_state=42
        )
        logger.info("⚠️  Using random split")
    
    train_landmarks = [landmarks[i] for i in train_idx]
    train_embeddings = embeddings[train_idx]
    train_labels = labels_onehot[train_idx]
    
    val_landmarks = [landmarks[i] for i in val_idx]
    val_embeddings = embeddings[val_idx]
    val_labels = labels_onehot[val_idx]
    
    logger.info(f"📊 Split: {len(train_landmarks)} train, {len(val_landmarks)} val")
    
    # Create generators
    train_gen = OptimizedDataGenerator(
        train_landmarks, train_embeddings, train_labels,
        batch_size=4, shuffle=True, augment=True
    )
    
    val_gen = OptimizedDataGenerator(
        val_landmarks, val_embeddings, val_labels,
        batch_size=4, shuffle=False, augment=False
    )
    
    # Build model
    model = build_optimized_fusion_model(num_classes)
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / f"gsl_final_{timestamp}_best.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=15,
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
        epochs=300,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save
    final_path = MODELS_DIR / f"gsl_final_{timestamp}.h5"
    model.save(str(final_path))
    
    # Save class mapping
    mapping_path = MODELS_DIR / f"class_mapping_{timestamp}.json"
    with open(mapping_path, 'w') as f:
        json.dump(class_to_idx, f, indent=2)
    
    # Results
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ Training Complete!")
    logger.info("=" * 80)
    logger.info(f"📊 Best Val Accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    logger.info(f"📊 Final Train Accuracy: {history.history['accuracy'][-1]:.4f}")
    logger.info(f"📊 Final Val Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    logger.info(f"📊 Number of classes: {num_classes}")
    logger.info(f"📊 Training samples: {len(train_landmarks)}")
    logger.info("=" * 80)
    
    return model, history, class_to_idx


if __name__ == "__main__":
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        model, history, class_mapping = train_final_model()
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        sys.exit(1)
