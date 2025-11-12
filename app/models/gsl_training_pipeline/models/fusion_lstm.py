"""
Fusion LSTM Model
Combines MediaPipe landmarks and ResNeXt embeddings for GSL recognition
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Dict
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def build_fusion_model(num_classes: int, config: Dict = None) -> Model:
    """
    Build fusion LSTM model combining landmarks and ResNeXt features
    
    Args:
        num_classes: Number of gesture classes
        config: Model configuration dict
    
    Returns:
        Compiled Keras model
    """
    if config is None:
        config = {
            'lstm_units': 256,
            'lstm_layers': 2,
            'dropout_rate': 0.4,
            'dense_units': [512, 256],
            'learning_rate': 0.001
        }
    
    logger.info("🏗️  Building Fusion LSTM Model...")
    
    # ========================================================================
    # LANDMARK BRANCH (Variable-length sequences)
    # ========================================================================
    landmark_input = layers.Input(shape=(None, 468), name='landmark_input')
    mask_input = layers.Input(shape=(None,), name='mask_input')
    
    # Bidirectional LSTM layers with masking
    x = landmark_input
    for i in range(config['lstm_layers']):
        x = layers.Bidirectional(
            layers.LSTM(
                config['lstm_units'],
                return_sequences=(i < config['lstm_layers'] - 1),
                dropout=config['dropout_rate'],
                recurrent_dropout=0.2
            ),
            name=f'bilstm_{i+1}'
        )(x, mask=mask_input)
        
        if i < config['lstm_layers'] - 1:
            x = layers.BatchNormalization()(x)
    
    # Global pooling for variable-length sequences
    landmark_features = layers.Dropout(config['dropout_rate'])(x)
    
    logger.info(f"   ✅ Landmark branch: BiLSTM({config['lstm_units']}) × {config['lstm_layers']}")
    
    # ========================================================================
    # RESNEXT BRANCH (Fixed-size embeddings)
    # ========================================================================
    resnext_input = layers.Input(shape=(512,), name='resnext_input')
    
    # Dense layers for ResNeXt features
    y = layers.Dense(256, activation='relu', name='resnext_dense1')(resnext_input)
    y = layers.Dropout(0.3)(y)
    resnext_features = layers.BatchNormalization()(y)
    
    logger.info(f"   ✅ ResNeXt branch: Dense(512→256)")
    
    # ========================================================================
    # FUSION LAYER
    # ========================================================================
    # Concatenate both branches
    fused = layers.Concatenate(name='fusion')([landmark_features, resnext_features])
    fused = layers.BatchNormalization()(fused)
    
    logger.info(f"   ✅ Fusion: Concatenate → {landmark_features.shape[-1] + resnext_features.shape[-1]} features")
    
    # ========================================================================
    # CLASSIFICATION HEAD
    # ========================================================================
    z = fused
    for idx, units in enumerate(config['dense_units']):
        z = layers.Dense(units, activation='relu', name=f'fc_{idx+1}')(z)
        z = layers.Dropout(config['dropout_rate'])(z)
        z = layers.BatchNormalization()(z)
    
    # Output layer
    output = layers.Dense(num_classes, activation='softmax', name='output')(z)
    
    logger.info(f"   ✅ Classification head: {' → '.join(map(str, config['dense_units']))} → {num_classes}")
    
    # ========================================================================
    # CREATE MODEL
    # ========================================================================
    model = Model(
        inputs=[landmark_input, resnext_input, mask_input],
        outputs=output,
        name='GSL_Fusion_LSTM'
    )
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    # Print summary
    total_params = model.count_params()
    logger.info(f"\n📊 Model Summary:")
    logger.info(f"   Total parameters: {total_params:,}")
    logger.info(f"   Trainable parameters: {total_params:,}")
    
    return model


def build_landmarks_only_model(num_classes: int, config: Dict = None) -> Model:
    """
    Build landmarks-only LSTM model (fallback if ResNeXt unavailable)
    
    Args:
        num_classes: Number of gesture classes
        config: Model configuration dict
    
    Returns:
        Compiled Keras model
    """
    if config is None:
        config = {
            'lstm_units': 256,
            'lstm_layers': 2,
            'dropout_rate': 0.4,
            'dense_units': [512, 256],
            'learning_rate': 0.001
        }
    
    logger.info("🏗️  Building Landmarks-Only LSTM Model...")
    
    # Input layers
    landmark_input = layers.Input(shape=(None, 468), name='landmark_input')
    mask_input = layers.Input(shape=(None,), name='mask_input')
    
    # Bidirectional LSTM layers
    x = landmark_input
    for i in range(config['lstm_layers']):
        x = layers.Bidirectional(
            layers.LSTM(
                config['lstm_units'],
                return_sequences=(i < config['lstm_layers'] - 1),
                dropout=config['dropout_rate'],
                recurrent_dropout=0.2
            ),
            name=f'bilstm_{i+1}'
        )(x, mask=mask_input)
        
        if i < config['lstm_layers'] - 1:
            x = layers.BatchNormalization()(x)
    
    x = layers.Dropout(config['dropout_rate'])(x)
    
    # Dense layers
    for idx, units in enumerate(config['dense_units']):
        x = layers.Dense(units, activation='relu', name=f'fc_{idx+1}')(x)
        x = layers.Dropout(config['dropout_rate'])(x)
        x = layers.BatchNormalization()(x)
    
    # Output layer
    output = layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    # Create model
    model = Model(
        inputs=[landmark_input, mask_input],
        outputs=output,
        name='GSL_Landmarks_LSTM'
    )
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    logger.info(f"📊 Total parameters: {model.count_params():,}")
    
    return model
