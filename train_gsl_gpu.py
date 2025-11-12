"""
Train GSL Model on NVIDIA GPU

This script trains the GSL fusion model using your NVIDIA GPU for faster training.
Includes automatic GPU detection and configuration.
"""

import sys
import os
import subprocess

print("=" * 70)
print("GSL MODEL TRAINING - GPU ACCELERATED")
print("=" * 70)

# Step 1: Check GPU availability
print("\n🔍 Checking GPU availability...")

try:
    import tensorflow as tf
    
    print(f"✓ TensorFlow version: {tf.__version__}")
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"✓ Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            # Enable memory growth to prevent TensorFlow from allocating all GPU memory
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  ✓ Memory growth enabled for GPU {i}")
            except RuntimeError as e:
                print(f"  ⚠️ Could not set memory growth: {e}")
        
        print("\n✅ GPU is ready for training!")
        use_gpu = True
    else:
        print("\n⚠️ No GPU detected!")
        print("\nTo enable GPU training:")
        print("1. Make sure you have NVIDIA GPU drivers installed")
        print("2. Install CUDA Toolkit (11.8 or 12.x)")
        print("3. Install cuDNN")
        print("4. Install tensorflow with GPU support:")
        print("   pip install tensorflow[and-cuda]")
        print("\nFor now, training will use CPU (slower).")
        
        response = input("\nContinue with CPU training? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            sys.exit(0)
        use_gpu = False

except ImportError:
    print("✗ TensorFlow not found!")
    print("Installing TensorFlow with GPU support...")
    subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow[and-cuda]"])
    print("\n✓ TensorFlow installed. Please restart the script.")
    sys.exit(0)

# Step 2: Import required modules
print("\n📦 Loading training modules...")

import json
import numpy as np
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers, Model
from datetime import datetime
from collections import Counter
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("✓ Modules loaded")

# Step 3: Load dataset
print("\n📂 Loading GSL dataset...")

def load_dataset():
    """Load the GSL dataset"""
    output_dir = Path("app/models/gsl_training_pipeline/output")
    
    # Load landmarks
    landmarks_file = output_dir / "gsl_landmarks.json"
    if not landmarks_file.exists():
        print(f"✗ Landmarks file not found: {landmarks_file}")
        print("\n💡 Please run feature extraction first:")
        print("   cd app/models")
        print("   python extract_features_complete.py")
        sys.exit(1)
    
    with open(landmarks_file, 'r', encoding='utf-8') as f:
        landmarks_data = json.load(f)
    
    # Load ResNeXt embeddings (if available)
    embeddings_file = output_dir / "resnext_embeddings.npy"
    if embeddings_file.exists():
        embeddings = np.load(embeddings_file)
        print(f"✓ Loaded ResNeXt embeddings: {embeddings.shape}")
    else:
        print("⚠️ ResNeXt embeddings not found, using landmarks only")
        # Create dummy embeddings
        embeddings = np.zeros((len(landmarks_data['clips']), 512))
    
    # Load class mapping
    class_index_file = output_dir / "class_index.json"
    with open(class_index_file, 'r', encoding='utf-8') as f:
        class_index_data = json.load(f)
    
    class_to_idx = {k: v for k, v in class_index_data.items() if k != 'num_classes'}
    
    # Extract data
    clips = landmarks_data['clips']
    landmarks = [np.array(c['landmarks'], dtype=np.float32) for c in clips]
    labels = [c['label'] for c in clips]
    
    print(f"✓ Loaded {len(landmarks)} clips")
    print(f"✓ Number of classes: {len(class_to_idx)}")
    print(f"✓ Classes: {', '.join(sorted(class_to_idx.keys()))}")
    
    return landmarks, embeddings, labels, class_to_idx

landmarks, embeddings, labels, class_to_idx = load_dataset()
num_classes = len(class_to_idx)

# Step 4: Build GPU-optimized model
print("\n🏗️ Building GPU-optimized model...")

def build_gpu_model(num_classes: int) -> Model:
    """Build model optimized for GPU training"""
    
    # Landmark input with masking
    landmark_input = layers.Input(shape=(None, 468), name='landmark_input')
    mask_input = layers.Input(shape=(None,), dtype='bool', name='mask_input')
    
    # Bidirectional LSTM (GPU-optimized)
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
    
    x_attended = layers.Multiply()([x, attention])
    landmark_features = layers.GlobalAveragePooling1D()(x_attended)
    landmark_features = layers.Dropout(0.4)(landmark_features)
    
    # ResNeXt branch
    resnext_input = layers.Input(shape=(512,), name='resnext_input')
    y = layers.Dense(256, activation='relu')(resnext_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.3)(y)
    y = layers.Dense(128, activation='relu')(y)
    y = layers.BatchNormalization()(y)
    resnext_features = layers.Dropout(0.3)(y)
    
    # Fusion
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
        name='GSL_GPU_Model'
    )
    
    # Use mixed precision for faster GPU training
    if use_gpu:
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print("✓ Mixed precision enabled (faster GPU training)")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_acc')]
    )
    
    print(f"✓ Model built with {model.count_params():,} parameters")
    
    return model

model = build_gpu_model(num_classes)

# Step 5: Prepare data
print("\n📊 Preparing training data...")

# Convert labels
label_indices = np.array([class_to_idx[label] for label in labels])
labels_onehot = keras.utils.to_categorical(label_indices, num_classes)

# Split data
indices = np.arange(len(landmarks))
train_idx, val_idx = train_test_split(indices, train_size=0.8, random_state=42)

print(f"✓ Training samples: {len(train_idx)}")
print(f"✓ Validation samples: {len(val_idx)}")

# Data generator
class GPUDataGenerator(keras.utils.Sequence):
    """GPU-optimized data generator"""
    
    def __init__(self, landmarks, embeddings, labels, indices, batch_size=16, shuffle=True, augment=False):
        self.landmarks = [landmarks[i] for i in indices]
        self.embeddings = embeddings[indices]
        self.labels = labels[indices]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(self.landmarks))
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
                # Add noise for augmentation
                noise = np.random.normal(0, 0.02, seq.shape)
                seq = seq + noise
                seq = np.clip(seq, 0, 1)
            
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

# Create generators with larger batch size for GPU
batch_size = 32 if use_gpu else 8

train_gen = GPUDataGenerator(
    landmarks, embeddings, labels_onehot, train_idx,
    batch_size=batch_size, shuffle=True, augment=True
)

val_gen = GPUDataGenerator(
    landmarks, embeddings, labels_onehot, val_idx,
    batch_size=batch_size, shuffle=False, augment=False
)

print(f"✓ Batch size: {batch_size} (optimized for {'GPU' if use_gpu else 'CPU'})")

# Step 6: Train on GPU
print("\n🚀 Starting GPU training...")
print("=" * 70)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
models_dir = Path("app/models/trained_models")
models_dir.mkdir(parents=True, exist_ok=True)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=str(models_dir / f"gsl_gpu_{timestamp}_best.h5"),
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
    ),
    keras.callbacks.TensorBoard(
        log_dir=f"logs/gpu_training_{timestamp}",
        histogram_freq=1
    )
]

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=100,
    callbacks=callbacks,
    verbose=1
)

# Save final model
final_path = models_dir / f"gsl_gpu_final_{timestamp}.h5"
model.save(str(final_path))

# Save class mapping
class_mapping_path = models_dir / f"class_mapping_{timestamp}.json"
with open(class_mapping_path, 'w') as f:
    json.dump(class_to_idx, f, indent=2)

# Results
print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)
print(f"📊 Best Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
print(f"📊 Best Validation Top-3: {max(history.history['val_top_3_acc']):.4f}")
print(f"📊 Final Train Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"📊 Final Val Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"\n💾 Models saved:")
print(f"   Best: {models_dir / f'gsl_gpu_{timestamp}_best.h5'}")
print(f"   Final: {final_path}")
print(f"   Class mapping: {class_mapping_path}")
print("=" * 70)

if use_gpu:
    print("\n🎉 Training completed on GPU!")
else:
    print("\n✓ Training completed on CPU")

print("\n💡 To use the trained model:")
print("   python live_sign_recognition.py")
