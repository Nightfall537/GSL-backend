# Requirements Document

## Introduction

This feature enables the GSL (Greek Sign Language) recognition system to train an LSTM model on variable-length gesture sequences extracted from segmented video clips. The system currently has 85 labeled gesture clips with varying durations (1-5 seconds), and needs to handle these sequences dynamically without forcing them to a fixed length, while preventing overfitting through proper regularization.

## Glossary

- **GSL_System**: The Greek Sign Language recognition system that processes video clips to identify sign language gestures
- **LSTM_Model**: Long Short-Term Memory neural network model used for sequence classification
- **Feature_Extractor**: Component that extracts MediaPipe Holistic landmarks (468 features) and ResNeXt-101 3D CNN features from video clips
- **Sequence_Padder**: Component that pads variable-length sequences to enable batch processing while preserving original sequence information
- **Training_Pipeline**: End-to-end process that loads features, prepares data, trains the model, and evaluates performance

## Requirements

### Requirement 1

**User Story:** As a GSL researcher, I want the system to handle variable-length gesture sequences, so that I can train on real gesture data without losing temporal information

#### Acceptance Criteria

1. WHEN the Feature_Extractor processes a video clip, THE GSL_System SHALL extract a sequence of landmarks with length equal to the number of frames in the clip
2. WHEN sequences have different lengths, THE Sequence_Padder SHALL pad shorter sequences with zeros to match the longest sequence in each batch
3. WHEN the LSTM_Model processes padded sequences, THE GSL_System SHALL use masking to ignore padded values during training
4. THE GSL_System SHALL preserve the original sequence length information for each sample
5. THE Training_Pipeline SHALL support batch sizes between 4 and 32 samples with mixed sequence lengths

### Requirement 2

**User Story:** As a machine learning engineer, I want the feature extraction to produce consistent feature dimensions, so that the LSTM model can process all clips without errors

#### Acceptance Criteria

1. THE Feature_Extractor SHALL extract exactly 468 landmark features per frame (132 pose + 63 left hand + 63 right hand + 210 face)
2. WHEN a frame lacks detected landmarks, THE Feature_Extractor SHALL fill missing values with zeros to maintain the 468-feature dimension
3. THE Feature_Extractor SHALL validate that each frame's feature vector has exactly 468 elements before adding to the sequence
4. THE GSL_System SHALL save extracted features in a format that preserves sequence structure (list of arrays with shape [num_frames, 468])
5. WHEN loading saved features, THE Training_Pipeline SHALL verify feature dimensions match the expected 468 features per frame

### Requirement 3

**User Story:** As a data scientist, I want the model to prevent overfitting on the small dataset, so that it generalizes well to new gesture samples

#### Acceptance Criteria

1. THE LSTM_Model SHALL include dropout layers with dropout rate between 0.3 and 0.5 after each LSTM layer
2. THE Training_Pipeline SHALL implement early stopping that monitors validation loss with patience of 10 epochs
3. THE Training_Pipeline SHALL split the dataset into 70% training and 30% validation sets with stratified sampling
4. THE LSTM_Model SHALL use L2 regularization with weight decay between 0.0001 and 0.001
5. THE Training_Pipeline SHALL apply data augmentation by adding Gaussian noise (std=0.01) to landmark coordinates during training

### Requirement 4

**User Story:** As a developer, I want the training pipeline to be robust and provide clear feedback, so that I can monitor progress and debug issues

#### Acceptance Criteria

1. THE Training_Pipeline SHALL log training metrics (loss, accuracy) for each epoch to the console
2. WHEN feature extraction fails for a clip, THE GSL_System SHALL log a warning with the clip name and continue processing remaining clips
3. THE Training_Pipeline SHALL save model checkpoints after each epoch that improves validation accuracy
4. THE GSL_System SHALL generate a training summary report with final accuracy, loss curves, and confusion matrix
5. WHEN the Training_Pipeline completes, THE GSL_System SHALL save the trained model weights to a file with timestamp

### Requirement 5

**User Story:** As a system integrator, I want the feature extraction to work with both MediaPipe landmarks and ResNeXt features, so that I can leverage both spatial and temporal information

#### Acceptance Criteria

1. THE Feature_Extractor SHALL attempt to extract ResNeXt-101 3D CNN features for each video clip
2. WHERE ResNeXt features are available, THE Feature_Extractor SHALL save them separately from landmark features
3. THE LSTM_Model SHALL support training with landmarks only when ResNeXt features are unavailable
4. WHERE both feature types exist, THE Training_Pipeline SHALL provide an option to train with fused features or landmarks only
5. THE GSL_System SHALL document which feature types were used for each trained model
