"""
Dynamic Sequence Data Generator
Handles variable-length sequences with padding and masking for LSTM training
"""
import numpy as np
import tensorflow as tf
from typing import List, Tuple, Dict
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class SequenceDataGenerator(tf.keras.utils.Sequence):
    """
    Data generator for variable-length sequences
    Implements dynamic padding and masking for batch processing
    """
    
    def __init__(self, 
                 landmarks: List[np.ndarray],
                 embeddings: np.ndarray,
                 labels: np.ndarray,
                 batch_size: int = 16,
                 shuffle: bool = True,
                 augment: bool = False,
                 noise_std: float = 0.01):
        """
        Initialize data generator
        
        Args:
            landmarks: List of landmark arrays [num_frames, 468]
            embeddings: ResNeXt embeddings [num_clips, 512]
            labels: One-hot encoded labels [num_clips, num_classes]
            batch_size: Batch size
            shuffle: Whether to shuffle data
            augment: Whether to apply data augmentation
            noise_std: Standard deviation for Gaussian noise augmentation
        """
        self.landmarks = landmarks
        self.embeddings = embeddings
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.noise_std = noise_std
        
        self.indices = np.arange(len(self.landmarks))
        self.on_epoch_end()
    
    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.landmarks) / self.batch_size))
    
    def __getitem__(self, idx):
        """
        Get one batch with dynamic padding
        
        Returns:
            Tuple of (inputs_dict, labels)
        """
        # Get batch indices
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        # Get sequences for this batch
        batch_landmarks = [self.landmarks[i] for i in batch_indices]
        batch_embeddings = self.embeddings[batch_indices]
        batch_labels = self.labels[batch_indices]
        
        # Find max length in this batch
        max_len = max(len(seq) for seq in batch_landmarks)
        
        # Pad sequences and create masks
        padded_landmarks = np.zeros((len(batch_landmarks), max_len, 468), dtype=np.float32)
        masks = np.zeros((len(batch_landmarks), max_len), dtype=np.float32)
        lengths = np.zeros(len(batch_landmarks), dtype=np.int32)
        
        for i, seq in enumerate(batch_landmarks):
            seq_len = len(seq)
            lengths[i] = seq_len
            
            # Apply augmentation if enabled
            if self.augment:
                seq = self._augment_sequence(seq)
            
            padded_landmarks[i, :seq_len] = seq
            masks[i, :seq_len] = 1.0  # Mark valid positions
        
        # Return as dictionary for multiple inputs
        inputs = {
            'landmark_input': padded_landmarks,
            'resnext_input': batch_embeddings,
            'mask_input': masks
        }
        
        return inputs, batch_labels
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _augment_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to sequence
        
        Args:
            sequence: Landmark sequence [num_frames, 468]
        
        Returns:
            Augmented sequence
        """
        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_std, sequence.shape)
        augmented = sequence + noise
        
        # Clip to valid range [0, 1] for normalized coordinates
        augmented = np.clip(augmented, 0, 1)
        
        return augmented.astype(np.float32)


def create_data_generators(landmarks: List[np.ndarray],
                           embeddings: np.ndarray,
                           labels: List[str],
                           class_to_idx: Dict[str, int],
                           train_split: float = 0.7,
                           batch_size: int = 16,
                           random_seed: int = 42) -> Tuple:
    """
    Create train and validation data generators
    
    Args:
        landmarks: List of landmark arrays
        embeddings: ResNeXt embeddings
        labels: List of label strings
        class_to_idx: Label to index mapping
        train_split: Training split ratio
        batch_size: Batch size
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_generator, val_generator, num_classes)
    """
    np.random.seed(random_seed)
    
    # Convert labels to indices
    label_indices = np.array([class_to_idx[label] for label in labels])
    num_classes = len(class_to_idx)
    
    # One-hot encode labels
    labels_onehot = tf.keras.utils.to_categorical(label_indices, num_classes)
    
    # Check if stratified split is possible
    from sklearn.model_selection import train_test_split
    from collections import Counter
    
    indices = np.arange(len(landmarks))
    label_counts = Counter(label_indices)
    min_samples = min(label_counts.values())
    
    # Use stratified split only if all classes have at least 2 samples
    if min_samples >= 2:
        logger.info("Using stratified split")
        train_idx, val_idx = train_test_split(
            indices,
            train_size=train_split,
            stratify=label_indices,
            random_state=random_seed
        )
    else:
        logger.warning(f"Some classes have only {min_samples} sample(s) - using random split")
        train_idx, val_idx = train_test_split(
            indices,
            train_size=train_split,
            random_state=random_seed
        )
    
    # Split data
    train_landmarks = [landmarks[i] for i in train_idx]
    train_embeddings = embeddings[train_idx]
    train_labels = labels_onehot[train_idx]
    
    val_landmarks = [landmarks[i] for i in val_idx]
    val_embeddings = embeddings[val_idx]
    val_labels = labels_onehot[val_idx]
    
    # Create generators
    train_gen = SequenceDataGenerator(
        train_landmarks,
        train_embeddings,
        train_labels,
        batch_size=batch_size,
        shuffle=True,
        augment=True  # Enable augmentation for training
    )
    
    val_gen = SequenceDataGenerator(
        val_landmarks,
        val_embeddings,
        val_labels,
        batch_size=batch_size,
        shuffle=False,
        augment=False  # No augmentation for validation
    )
    
    logger.info(f"📊 Data split:")
    logger.info(f"   Training: {len(train_landmarks)} clips")
    logger.info(f"   Validation: {len(val_landmarks)} clips")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Classes: {num_classes}")
    
    return train_gen, val_gen, num_classes
