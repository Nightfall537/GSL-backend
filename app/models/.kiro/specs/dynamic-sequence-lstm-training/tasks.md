# Implementation Plan

## Overview
This implementation plan breaks down the GSL LSTM training pipeline into discrete, actionable coding tasks. Each task builds incrementally on previous work, following the design document's architecture. The focus is on creating a fully trained, highly functional, and accurate model using both MediaPipe landmarks and ResNeXt-101 3D CNN features.

**Current Status**: 85 manually labeled clips ready in `segmented_clips/` with `segmentation_metadata.json`

**Goal**: Complete Stages 3-7 of the pipeline before proceeding to TTS/STT integration

---

## Tasks

- [x] 1. Setup project structure and dependencies


  - Create directory structure for feature extraction, training, and models
  - Install required dependencies (TensorFlow, PyTorch, MediaPipe, OpenCV)
  - Create configuration file for training parameters
  - Setup logging infrastructure
  - _Requirements: 4.1, 4.2, 4.5_




- [ ] 2. Implement MediaPipe Holistic feature extraction (Stage 3)
  - [ ] 2.1 Create MediaPipeFeatureExtractor class
    - Initialize MediaPipe Holistic with model_complexity=2
    - Implement extract_landmarks() method for single video
    - Extract 543 landmarks: pose(33×4) + hands(42×3) + face(468×3)

    - Handle missing landmarks by filling with zeros
    - _Requirements: 2.1, 2.2, 2.3_
  
  - [ ] 2.2 Implement landmark validation and flattening
    - Create validate_feature_dimensions() to ensure 543 features per frame

    - Implement flatten_landmarks() to convert dict to [N_frames, 543] array
    - Add frame-level validation before appending to sequence
    - _Requirements: 2.3, 2.4_
  
  - [ ] 2.3 Batch process all 85 clips
    - Load segmentation_metadata.json to get labeled clips

    - Process each clip and extract landmarks
    - Log warnings for failed extractions and continue
    - Store results with clip metadata (name, label, num_frames)
    - _Requirements: 2.5, 4.2_
  



  - [ ] 2.4 Save landmarks to gsl_landmarks.json
    - Create JSON structure with clips array and metadata
    - Include clip_name, label, landmarks array, num_frames, duration
    - Add dataset metadata (total_clips, landmark_dimension, extraction_date)
    - Validate JSON file integrity after saving
    - _Requirements: 2.4, 2.5_


- [ ] 3. Implement ResNeXt-101 3D CNN feature extraction (Stage 4 - CRITICAL)
  - [ ] 3.1 Create ResNeXt3DFeatureExtractor class
    - Initialize PyTorch and check for GPU availability
    - Load pre-trained R3D-18 or MC3-18 model (Kinetics-400)
    - Remove classification head to extract embeddings
    - Set model to eval mode and move to appropriate device

    - _Requirements: 5.1, 5.2_
  
  - [ ] 3.2 Implement video preprocessing for 3D CNN
    - Create preprocess_video() to load and sample 16 frames uniformly
    - Resize frames to 112×112 resolution
    - Handle clips with < 16 frames by repeating last frame
    - Normalize with ImageNet statistics

    - Convert to tensor format [1, 3, 16, 112, 112]
    - _Requirements: 5.1_
  
  - [ ] 3.3 Implement feature extraction with error handling
    - Create extract_features() method for single video
    - Use torch.no_grad() for efficient inference

    - Extract 512-dimensional embedding vector
    - Handle GPU OOM errors by falling back to CPU
    - Return zero embedding on failure with warning log
    - _Requirements: 5.1, 5.2, 4.2_
  
  - [ ] 3.4 Batch extract features for all 85 clips
    - Process all clips from segmentation_metadata.json
    - Store embeddings in list maintaining clip order
    - Log progress and any extraction failures
    - Validate all embeddings have shape [512]
    - _Requirements: 5.2, 4.2_
  
  - [ ] 3.5 Save embeddings to resnext_embeddings.npy
    - Convert list to NumPy array [85, 512]
    - Save using np.save() for efficient loading
    - Verify file integrity after saving
    - _Requirements: 5.2_

- [x] 4. Implement dataset packaging module (Stage 5)


  - [ ] 4.1 Create GSLDatasetPackager class
    - Load segmentation_metadata.json
    - Extract unique labels from all clips
    - Create class_index mapping (label → integer)
    - Validate label consistency across files
    - _Requirements: 2.5, 5.5_
  
  - [ ] 4.2 Create class_index.json
    - Generate sorted label list from metadata
    - Create dictionary mapping labels to indices
    - Add num_classes field
    - Save to JSON file
    - _Requirements: 2.5_
  
  - [ ] 4.3 Create dataset_info.json
    - Compile metadata: total_clips, num_classes, landmark_dimension
    - Add extraction dates and file paths
    - Include per-clip information (frames, duration, label)
    - Document ResNeXt embedding availability
    - _Requirements: 4.5_
  
  - [ ] 4.4 Implement dataset validation
    - Verify all clips have both landmarks and embeddings
    - Check dimension consistency (543 for landmarks, 512 for embeddings)
    - Validate label matching across all files
    - Report any missing or corrupted data
    - _Requirements: 2.5, 5.2_

- [ ] 5. Implement dynamic sequence handling for variable-length clips
  - [ ] 5.1 Create SequenceDataGenerator class
    - Inherit from tf.keras.utils.Sequence
    - Store landmarks (variable length), embeddings (fixed), labels
    - Implement __len__() and __getitem__() methods
    - Support shuffling between epochs
    - _Requirements: 1.1, 1.2, 1.5_
  
  - [ ] 5.2 Implement dynamic padding in collate function
    - Find maximum sequence length within each batch
    - Pad shorter sequences with zeros to match max length
    - Create boolean masks: 1 for real frames, 0 for padding
    - Return dict with padded_landmarks, embeddings, masks, labels
    - _Requirements: 1.2, 1.3_
  
  - [ ] 5.3 Implement data augmentation
    - Add Gaussian noise (std=0.01) to landmark coordinates during training
    - Apply augmentation only to training set, not validation
    - Ensure augmentation preserves sequence structure
    - _Requirements: 3.5_
  
  - [ ] 5.4 Create train/validation split
    - Implement stratified split (70% train, 30% validation)
    - Ensure balanced class distribution in both sets
    - Store split indices for reproducibility
    - _Requirements: 3.3_

- [ ] 6. Build LSTM model architectures (Stage 6)
  - [ ] 6.1 Implement landmarks-only LSTM model
    - Create Input layer for landmarks [None, 543]
    - Add masking layer to handle variable lengths
    - Stack 2 bidirectional LSTM layers (256 units each)
    - Add Dropout(0.4) and BatchNormalization
    - Add Dense layers (512→256→num_classes) with activations
    - Compile with Adam optimizer and categorical crossentropy
    - _Requirements: 3.1, 3.4_
  
  - [ ] 6.2 Implement fusion LSTM model (RECOMMENDED)
    - Create landmark input branch with masked bidirectional LSTM
    - Create ResNeXt input branch with Dense layers
    - Concatenate both branches after global pooling
    - Add fusion layers: Dense(768→512→256) with dropout
    - Add final classification layer with softmax
    - Compile with appropriate loss and metrics
    - _Requirements: 5.3, 5.4, 3.1, 3.4_
  
  - [ ] 6.3 Implement model builder with configuration
    - Create build_model() function accepting config dict
    - Support both 'landmarks_only' and 'fusion' model types
    - Allow configurable LSTM units, layers, dropout rates
    - Return compiled model ready for training
    - _Requirements: 3.1, 3.4_

- [ ] 7. Implement training pipeline with optimizations
  - [ ] 7.1 Setup training callbacks
    - Create EarlyStopping callback (patience=15, monitor='val_loss')
    - Create ModelCheckpoint callback (save best model by val_accuracy)
    - Create ReduceLROnPlateau callback (factor=0.5, patience=5)
    - Create TensorBoard callback for visualization
    - _Requirements: 3.2, 4.3_
  
  - [ ] 7.2 Implement K-fold cross validation
    - Use StratifiedKFold with k=5 folds
    - Train separate model for each fold
    - Track per-fold metrics (accuracy, loss)
    - Calculate mean and std of cross-validation scores
    - _Requirements: 3.3_
  
  - [ ] 7.3 Create main training loop
    - Load prepared dataset (landmarks, embeddings, labels)
    - Create data generators with dynamic padding
    - Build model based on configuration
    - Train with callbacks and validation data
    - Save best model weights and training history
    - _Requirements: 3.2, 4.1, 4.3_
  
  - [ ] 7.4 Implement training with error recovery
    - Handle KeyboardInterrupt to save checkpoint
    - Catch and log training exceptions
    - Save interrupted model for recovery
    - Support resuming from checkpoint
    - _Requirements: 4.2, 4.3_

- [ ] 8. Implement model evaluation and reporting
  - [ ] 8.1 Create evaluation metrics calculator
    - Calculate accuracy, precision, recall, F1-score per class
    - Generate confusion matrix
    - Calculate top-3 accuracy
    - Compute per-class performance metrics
    - _Requirements: 4.1, 4.4_
  
  - [ ] 8.2 Generate training report
    - Create comprehensive evaluation report with all metrics
    - Include training curves (loss and accuracy over epochs)
    - Add confusion matrix visualization
    - Document model architecture and parameters
    - Save report as JSON and generate PDF/HTML visualization
    - _Requirements: 4.4, 4.5_
  
  - [ ] 8.3 Implement model persistence
    - Save final model to gsl_lstm_model.h5 or resnext_lstm_model.h5
    - Save model configuration and class_index together
    - Create model loading utility for inference
    - Add timestamp to saved models
    - _Requirements: 4.5_

- [ ] 9. Create complete training script
  - [ ] 9.1 Implement end-to-end training pipeline
    - Create main script that orchestrates all stages
    - Load configuration from config file
    - Run feature extraction (MediaPipe + ResNeXt)
    - Package dataset files
    - Train fusion model with all optimizations
    - Generate evaluation report
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1-2.5, 3.1-3.5, 4.1-4.5, 5.1-5.5_
  
  - [ ] 9.2 Add command-line interface
    - Support arguments for config file path
    - Add flags for model type (landmarks_only vs fusion)
    - Support resume from checkpoint
    - Add verbose logging option
    - _Requirements: 4.1, 4.2_
  
  - [ ] 9.3 Create configuration file template
    - Define all training hyperparameters
    - Include paths for data and output directories
    - Document each configuration option
    - Provide sensible defaults
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 10. Implement real-time inference system (Stage 7)
  - [ ] 10.1 Create real-time gesture recognizer
    - Load trained fusion model
    - Initialize MediaPipe Holistic for live processing
    - Capture webcam frames using OpenCV
    - Extract landmarks from each frame
    - Maintain sliding window of recent frames
    - _Requirements: 1.1, 1.4_
  
  - [ ] 10.2 Implement gesture detection logic
    - Detect gesture start/end based on motion
    - Buffer frames during gesture performance
    - Extract features when gesture completes
    - Run model inference on buffered sequence
    - Display predicted label with confidence
    - _Requirements: 1.4_
  
  - [ ] 10.3 Optimize for latency target
    - Ensure inference completes in < 80ms per frame
    - Use model optimization techniques (quantization if needed)
    - Implement frame skipping if necessary
    - Profile and optimize bottlenecks
    - _Requirements: 1.4_
  
  - [ ] 10.4 Create visualization interface
    - Draw landmarks on video feed
    - Display predicted gesture label
    - Show confidence scores
    - Add FPS counter
    - _Requirements: 4.1_

- [ ] 11. Validation and testing
  - [ ] 11.1 Validate feature extraction
    - Test MediaPipe extraction on all 85 clips
    - Verify 543 dimensions per frame
    - Test ResNeXt extraction on all clips
    - Verify 512-dimensional embeddings
    - Ensure no NaN or infinite values
    - _Requirements: 2.1, 2.2, 2.3, 5.1, 5.2_
  
  - [ ] 11.2 Validate model training
    - Verify model trains without errors
    - Check validation accuracy > 85%
    - Ensure early stopping triggers appropriately
    - Validate model checkpoints save correctly
    - Test model loading and inference
    - _Requirements: 3.1, 3.2, 3.3, 4.3_
  
  - [ ] 11.3 Validate real-time performance
    - Test webcam capture and processing
    - Measure inference latency (target < 80ms)
    - Verify gesture predictions are accurate
    - Test on various lighting conditions
    - _Requirements: 1.4_
  
  - [ ] 11.4 Generate final validation report
    - Document all validation results
    - Include performance metrics and benchmarks
    - List any issues or limitations discovered
    - Confirm all success criteria are met
    - _Requirements: 4.4, 4.5_

---

## Success Criteria

Before proceeding to TTS/STT integration (Stages 8-9), the following must be achieved:

1. ✅ All 85 clips successfully processed for MediaPipe landmarks
2. ✅ All 85 clips successfully processed for ResNeXt embeddings
3. ✅ Fusion model trained with validation accuracy > 85%
4. ✅ Inference latency < 80ms per frame
5. ✅ Model saves and loads correctly
6. ✅ Confusion matrix shows good class separation
7. ✅ Real-time webcam demo works smoothly

**The model must be fully trained, highly functional, and accurate before moving forward.**
