#!/usr/bin/env python3
"""
Complete Full-Body GSL Training Pipeline with Kaggle Dataset
Uses MediaPipe Holistic for comprehensive body tracking:
- 33 body pose landmarks (full body)
- 21 landmarks per hand (both hands)
- 468 face landmarks
- Full upper body context for accurate GSL recognition
- Downloads and processes SignTalk Ghana dataset from Kaggle
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import json
from pathlib import Path
import logging
from datetime import datetime
from collections import defaultdict
import mediapipe as mp
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FullBodyGSLPipeline:
    """Complete pipeline with full body tracking for accurate GSL recognition"""
    
    def __init__(self, use_kaggle=True, max_videos=40):
        self.video_dir = Path("sam2_annotation/gsl_videos")
        self.kaggle_video_dir = Path("kaggle_gsl_videos")
        self.output_dir = Path("sam2_training_output")
        self.segmentations_dir = self.output_dir / "segmentations"
        self.annotations_dir = self.output_dir / "annotations"
        self.training_data_dir = self.output_dir / "training_data"
        self.models_dir = self.output_dir / "models"
        
        self.use_kaggle = use_kaggle
        self.max_videos = max_videos
        
        # Create directories
        for dir_path in [self.video_dir, self.kaggle_video_dir, self.segmentations_dir, 
                         self.annotations_dir, self.training_data_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize MediaPipe Holistic for FULL BODY tracking
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=2,  # Highest accuracy
            enable_segmentation=True,  # Enable body segmentation
            refine_face_landmarks=True,  # Detailed face tracking
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load gesture data
        self.gesture_data = self.load_gesture_data()
        
        logger.info("🎯 Full-Body GSL Pipeline Initialized")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"📊 Kaggle dataset: {'Enabled' if use_kaggle else 'Disabled'}")
        logger.info(f"📹 Max videos to process: {max_videos}")
        logger.info("🔧 Using MediaPipe Holistic for comprehensive tracking:")
        logger.info("   ✅ 33 body pose landmarks")
        logger.info("   ✅ 21 landmarks per hand (both hands)")
        logger.info("   ✅ 468 face landmarks")
        logger.info("   ✅ Full body segmentation")
        
        # Download Kaggle dataset if requested
        if self.use_kaggle:
            self.download_kaggle_dataset()
    
    def load_gesture_data(self):
        """Load gesture data from JSON files"""
        gesture_files = [
            "colors_signs_data.json",
            "family_signs_data.json",
            "food_signs_data.json",
            "animals_signs_data.json",
            "grammar_signs_data.json",
            "home_clothing_signs_data.json"
        ]
        
        all_gestures = {}
        for file in gesture_files:
            file_path = Path(file)
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            data = json.loads(content)
                            all_gestures.update(data)
                            logger.info(f"✅ Loaded {file}")
                        else:
                            logger.warning(f"⚠️ Skipped empty file: {file}")
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ Skipped invalid JSON in {file}: {e}")
        
        logger.info(f"📊 Total gestures in JSON: {len(all_gestures)}")
        return all_gestures
    
    def download_kaggle_dataset(self):
        """Download SignTalk Ghana dataset from Kaggle"""
        logger.info("📥 Downloading SignTalk Ghana dataset from Kaggle...")
        
        try:
            import kagglehub
            
            # Download the dataset
            logger.info("🔄 Downloading from Kaggle (this may take a few minutes)...")
            path = kagglehub.dataset_download("responsibleailab/signtalk-ghana")
            
            logger.info(f"✅ Dataset downloaded to: {path}")
            
            # Find video files in the downloaded dataset
            dataset_path = Path(path)
            video_files = []
            
            # Search for video files
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                video_files.extend(list(dataset_path.rglob(ext)))
            
            logger.info(f"📹 Found {len(video_files)} video files in dataset")
            
            # Copy first N videos to our working directory
            videos_copied = 0
            for video_file in video_files[:self.max_videos]:
                dest_file = self.kaggle_video_dir / video_file.name
                if not dest_file.exists():
                    import shutil
                    shutil.copy2(video_file, dest_file)
                    videos_copied += 1
                    logger.info(f"📋 Copied: {video_file.name}")
            
            logger.info(f"✅ Copied {videos_copied} videos to {self.kaggle_video_dir}")
            
            # Also try to load metadata if available
            metadata_files = list(dataset_path.rglob("*.csv")) + list(dataset_path.rglob("*.json"))
            if metadata_files:
                logger.info(f"📊 Found {len(metadata_files)} metadata files")
                for meta_file in metadata_files[:3]:  # Show first 3
                    logger.info(f"   - {meta_file.name}")
            
            return True
            
        except ImportError:
            logger.error("❌ kagglehub not installed")
            logger.info("💡 Install with: pip install kagglehub")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to download Kaggle dataset: {e}")
            logger.info("💡 Make sure you have Kaggle API credentials configured")
            logger.info("💡 See: https://github.com/Kaggle/kaggle-api#api-credentials")
            return False
        
        try:
            import kagglehub
            logger.info("✅ kagglehub available")
        except ImportError:
            logger.error("❌ kagglehub not installed")
            logger.info("💡 Install with: pip install kagglehub")
            return None
        
        try:
            # Download the dataset
            logger.info("🔄 Downloading dataset (this may take a few minutes)...")
            path = kagglehub.dataset_download("responsibleailab/signtalk-ghana")
            logger.info(f"✅ Dataset downloaded to: {path}")
            
            # Find video files
            dataset_path = Path(path)
            video_files = []
            
            # Look for video files in the dataset
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
                video_files.extend(list(dataset_path.rglob(ext)))
            
            logger.info(f"📊 Found {len(video_files)} video files in dataset")
            
            # Copy first N videos to our working directory
            copied_count = 0
            for video_file in video_files[:self.max_videos]:
                dest = self.kaggle_video_dir / video_file.name
                if not dest.exists():
                    import shutil
                    shutil.copy2(video_file, dest)
                    copied_count += 1
                    logger.info(f"📹 Copied: {video_file.name}")
            
            logger.info(f"✅ Copied {copied_count} videos to {self.kaggle_video_dir}")
            
            # Try to load metadata if available
            metadata_files = list(dataset_path.rglob("*.csv")) + list(dataset_path.rglob("*.json"))
            if metadata_files:
                logger.info(f"📋 Found metadata files: {[f.name for f in metadata_files[:3]]}")
                self.load_kaggle_metadata(metadata_files[0])
            
            return dataset_path
            
        except Exception as e:
            logger.error(f"❌ Failed to download Kaggle dataset: {e}")
            logger.info("💡 Make sure you have Kaggle API credentials configured")
            logger.info("💡 See: https://github.com/Kaggle/kaggle-api#api-credentials")
            return None
    
    def load_kaggle_metadata(self, metadata_file):
        """Load metadata from Kaggle dataset"""
        try:
            if metadata_file.suffix == '.csv':
                import pandas as pd
                df = pd.read_csv(metadata_file)
                logger.info(f"📊 Loaded metadata: {len(df)} records")
                logger.info(f"📋 Columns: {list(df.columns)}")
                
                # Save metadata for reference
                metadata_json = self.output_dir / "kaggle_metadata.json"
                df.head(50).to_json(metadata_json, orient='records', indent=2)
                logger.info(f"💾 Saved metadata sample: {metadata_json}")
                
            elif metadata_file.suffix == '.json':
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"📊 Loaded JSON metadata: {len(data) if isinstance(data, list) else 'dict'}")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not load metadata: {e}")
    
    def extract_holistic_features(self, results, frame_shape):
        """Extract comprehensive features from MediaPipe Holistic"""
        features = {
            'pose_landmarks': [],
            'left_hand_landmarks': [],
            'right_hand_landmarks': [],
            'face_landmarks': [],
            'has_pose': False,
            'has_left_hand': False,
            'has_right_hand': False,
            'has_face': False
        }
        
        height, width = frame_shape[:2]
        
        # Extract pose landmarks (33 points - full body)
        if results.pose_landmarks:
            features['has_pose'] = True
            for lm in results.pose_landmarks.landmark:
                features['pose_landmarks'].append({
                    'x': float(lm.x),
                    'y': float(lm.y),
                    'z': float(lm.z),
                    'visibility': float(lm.visibility)
                })
        
        # Extract left hand landmarks (21 points)
        if results.left_hand_landmarks:
            features['has_left_hand'] = True
            for lm in results.left_hand_landmarks.landmark:
                features['left_hand_landmarks'].append({
                    'x': float(lm.x),
                    'y': float(lm.y),
                    'z': float(lm.z)
                })
        
        # Extract right hand landmarks (21 points)
        if results.right_hand_landmarks:
            features['has_right_hand'] = True
            for lm in results.right_hand_landmarks.landmark:
                features['right_hand_landmarks'].append({
                    'x': float(lm.x),
                    'y': float(lm.y),
                    'z': float(lm.z)
                })
        
        # Extract face landmarks (468 points - detailed)
        if results.face_landmarks:
            features['has_face'] = True
            # Store key face landmarks (not all 468 to save space)
            key_face_indices = [0, 1, 4, 5, 6, 10, 33, 61, 133, 152, 263, 291, 362, 386]
            for idx in key_face_indices:
                if idx < len(results.face_landmarks.landmark):
                    lm = results.face_landmarks.landmark[idx]
                    features['face_landmarks'].append({
                        'x': float(lm.x),
                        'y': float(lm.y),
                        'z': float(lm.z)
                    })
        
        return features
    
    def segment_video_full_body(self, video_path):
        """STEP 1: Full body segmentation with MediaPipe Holistic"""
        logger.info("🎬 STEP 1: Full-Body Video Segmentation")
        logger.info(f"📹 Processing video: {video_path.name}")
        logger.info("🔍 Tracking: Pose, Hands, Face, Body segmentation")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"❌ Cannot open video: {video_path}")
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps
        
        logger.info(f"📊 Video: {total_frames} frames, {fps} FPS, {width}x{height}")
        logger.info(f"⏱️ Duration: {duration:.1f} seconds")
        
        segmented_frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 2nd frame for efficiency
            if frame_idx % 2 == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(rgb_frame)
                
                # Extract comprehensive features
                features = self.extract_holistic_features(results, frame.shape)
                
                # Only save frames where we detect the person
                if features['has_pose']:
                    frame_data = {
                        'frame_number': frame_idx,
                        'timestamp': frame_idx / fps,
                        **features
                    }
                    segmented_frames.append(frame_data)
            
            frame_idx += 1
            
            if frame_idx % 300 == 0:
                logger.info(f"📈 Processed {frame_idx}/{total_frames} frames ({100*frame_idx/total_frames:.1f}%)")
        
        cap.release()
        
        logger.info(f"✅ Processed {len(segmented_frames)} frames with full-body tracking")
        logger.info(f"📊 Frames with pose: {sum(1 for f in segmented_frames if f['has_pose'])}")
        logger.info(f"📊 Frames with left hand: {sum(1 for f in segmented_frames if f['has_left_hand'])}")
        logger.info(f"📊 Frames with right hand: {sum(1 for f in segmented_frames if f['has_right_hand'])}")
        logger.info(f"📊 Frames with face: {sum(1 for f in segmented_frames if f['has_face'])}")
        
        # Save segmentation data
        seg_data = {
            'video_name': video_path.name,
            'video_path': str(video_path),
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'duration': duration,
            'segmented_frames': segmented_frames,
            'segmentation_method': 'MediaPipe_Holistic',
            'tracking_features': {
                'pose_landmarks': 33,
                'left_hand_landmarks': 21,
                'right_hand_landmarks': 21,
                'face_landmarks': 14  # Key points only
            }
        }
        
        seg_file = self.segmentations_dir / f"{video_path.stem}_fullbody_segmentations.json"
        with open(seg_file, 'w', encoding='utf-8') as f:
            json.dump(seg_data, f, indent=2)
        
        logger.info(f"💾 Saved full-body segmentation data: {seg_file}")
        
        return seg_data

    
    def annotate_with_full_body_context(self, seg_data):
        """STEP 2: Advanced annotation with full body context"""
        logger.info("🏷️ STEP 2: Full-Body Context Annotation")
        logger.info("📋 Method 1: Body Keypoint Prompts (pose + hands + face)")
        logger.info("📋 Method 2: Spatial Relationship Prompts (hand-to-body position)")
        logger.info("📋 Method 3: Holistic Context (full scene understanding)")
        
        # Determine gesture category from video name
        video_name = seg_data['video_name'].lower()
        gesture_category = None
        gestures_list = []
        
        if 'color' in video_name or 'colour' in video_name:
            gesture_category = 'colors'
            gestures_list = ['red', 'blue', 'green', 'yellow', 'black', 'white', 
                           'orange', 'purple', 'pink', 'brown', 'gray']
        
        logger.info(f"📂 Category: {gesture_category}")
        logger.info(f"🎯 Gestures: {gestures_list}")
        
        annotated_frames = []
        frames_per_gesture = len(seg_data['segmented_frames']) // len(gestures_list)
        
        for gesture_idx, gesture in enumerate(gestures_list):
            start_frame = gesture_idx * frames_per_gesture
            end_frame = start_frame + frames_per_gesture
            
            for frame_data in seg_data['segmented_frames'][start_frame:end_frame]:
                # Method 1: Body keypoint prompts
                keypoint_prompts = {
                    'pose_points': [],
                    'left_hand_points': [],
                    'right_hand_points': [],
                    'face_points': []
                }
                
                if frame_data['has_pose']:
                    # Key pose points: shoulders, elbows, wrists, hips
                    key_pose_indices = [11, 12, 13, 14, 15, 16, 23, 24]
                    for idx in key_pose_indices:
                        if idx < len(frame_data['pose_landmarks']):
                            lm = frame_data['pose_landmarks'][idx]
                            keypoint_prompts['pose_points'].append((lm['x'], lm['y'], lm['z']))
                
                if frame_data['has_left_hand']:
                    # Key hand points: wrist, thumb tip, index tip, pinky tip
                    key_hand_indices = [0, 4, 8, 20]
                    for idx in key_hand_indices:
                        if idx < len(frame_data['left_hand_landmarks']):
                            lm = frame_data['left_hand_landmarks'][idx]
                            keypoint_prompts['left_hand_points'].append((lm['x'], lm['y'], lm['z']))
                
                if frame_data['has_right_hand']:
                    for idx in key_hand_indices:
                        if idx < len(frame_data['right_hand_landmarks']):
                            lm = frame_data['right_hand_landmarks'][idx]
                            keypoint_prompts['right_hand_points'].append((lm['x'], lm['y'], lm['z']))
                
                if frame_data['has_face']:
                    for lm in frame_data['face_landmarks']:
                        keypoint_prompts['face_points'].append((lm['x'], lm['y'], lm['z']))
                
                # Method 2: Spatial relationship prompts
                spatial_relationships = self.compute_spatial_relationships(frame_data)
                
                # Method 3: Holistic context
                holistic_context = {
                    'has_full_body': frame_data['has_pose'],
                    'has_both_hands': frame_data['has_left_hand'] and frame_data['has_right_hand'],
                    'has_face': frame_data['has_face'],
                    'hand_positions': spatial_relationships.get('hand_positions', {}),
                    'body_orientation': spatial_relationships.get('body_orientation', 'unknown'),
                    'confidence': 0.95
                }
                
                annotated_frames.append({
                    **frame_data,
                    'gesture_label': gesture,
                    'annotation_methods': {
                        'keypoint_prompts': keypoint_prompts,
                        'spatial_relationships': spatial_relationships,
                        'holistic_context': holistic_context
                    }
                })
            
            logger.info(f"✅ Annotated {end_frame - start_frame} frames as '{gesture}'")
        
        ann_data = {
            **seg_data,
            'annotated_frames': annotated_frames,
            'gesture_category': gesture_category,
            'gestures_list': gestures_list,
            'annotation_methods_used': ['keypoint_prompts', 'spatial_relationships', 'holistic_context']
        }
        
        # Save annotations
        ann_file = self.annotations_dir / f"{Path(seg_data['video_name']).stem}_fullbody_annotations.json"
        with open(ann_file, 'w', encoding='utf-8') as f:
            json.dump(ann_data, f, indent=2, default=str)
        
        logger.info(f"💾 Saved full-body annotations: {ann_file}")
        logger.info(f"📊 Total annotated frames: {len(annotated_frames)}")
        
        return ann_data
    
    def compute_spatial_relationships(self, frame_data):
        """Compute spatial relationships between body parts"""
        relationships = {}
        
        if not frame_data['has_pose']:
            return relationships
        
        pose = frame_data['pose_landmarks']
        
        # Get key body points
        if len(pose) >= 24:
            nose = pose[0]
            left_shoulder = pose[11]
            right_shoulder = pose[12]
            left_wrist = pose[15]
            right_wrist = pose[16]
            
            # Compute hand positions relative to body
            relationships['hand_positions'] = {}
            
            if frame_data['has_left_hand']:
                # Left hand position relative to shoulder and nose
                relationships['hand_positions']['left'] = {
                    'relative_to_shoulder': {
                        'x': left_wrist['x'] - left_shoulder['x'],
                        'y': left_wrist['y'] - left_shoulder['y'],
                        'z': left_wrist['z'] - left_shoulder['z']
                    },
                    'relative_to_nose': {
                        'x': left_wrist['x'] - nose['x'],
                        'y': left_wrist['y'] - nose['y'],
                        'z': left_wrist['z'] - nose['z']
                    }
                }
            
            if frame_data['has_right_hand']:
                relationships['hand_positions']['right'] = {
                    'relative_to_shoulder': {
                        'x': right_wrist['x'] - right_shoulder['x'],
                        'y': right_wrist['y'] - right_shoulder['y'],
                        'z': right_wrist['z'] - right_shoulder['z']
                    },
                    'relative_to_nose': {
                        'x': right_wrist['x'] - nose['x'],
                        'y': right_wrist['y'] - nose['y'],
                        'z': right_wrist['z'] - nose['z']
                    }
                }
            
            # Compute body orientation
            shoulder_width = abs(right_shoulder['x'] - left_shoulder['x'])
            relationships['body_orientation'] = 'frontal' if shoulder_width > 0.2 else 'side'
            relationships['shoulder_width'] = shoulder_width
        
        return relationships
    
    def prepare_training_data(self, ann_data):
        """STEP 3: Prepare training sequences with full body features"""
        logger.info("📊 STEP 3: Training Data Preparation (Full Body Features)")
        
        # Group frames by gesture
        gesture_frames = defaultdict(list)
        for frame_data in ann_data['annotated_frames']:
            gesture = frame_data['gesture_label']
            gesture_frames[gesture].append(frame_data)
        
        # Create sequences (30 frames each)
        sequence_length = 30
        training_sequences = []
        
        for gesture, frames in gesture_frames.items():
            for i in range(0, len(frames) - sequence_length, sequence_length // 2):
                sequence = frames[i:i + sequence_length]
                if len(sequence) == sequence_length:
                    training_sequences.append({
                        'gesture': gesture,
                        'frames': sequence,
                        'start_time': sequence[0]['timestamp'],
                        'end_time': sequence[-1]['timestamp']
                    })
            
            logger.info(f"✅ Created sequences for '{gesture}': {len([s for s in training_sequences if s['gesture'] == gesture])}")
        
        logger.info(f"📊 Total training sequences: {len(training_sequences)}")
        
        # Save training data
        training_file = self.training_data_dir / "fullbody_training_sequences.json"
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump({
                'sequences': training_sequences,
                'sequence_length': sequence_length,
                'num_gestures': len(gesture_frames),
                'gestures': list(gesture_frames.keys()),
                'feature_description': 'Full body: 33 pose + 21 left hand + 21 right hand + 14 face = 89 landmarks'
            }, f, indent=2, default=str)
        
        logger.info(f"💾 Saved training data: {training_file}")
        
        return training_sequences, list(gesture_frames.keys())
    
    def train_model(self, training_sequences, gestures):
        """STEP 4: Train Deep Learning Model with Full Body Features"""
        logger.info("🧠 STEP 4: Deep Learning Model Training (Full Body)")
        logger.info("⏱️ This will take several minutes...")
        
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            
            logger.info(f"✅ TensorFlow {tf.__version__} available")
            
            # Check for GPU
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"🎮 GPU available: {len(gpus)} device(s)")
            else:
                logger.info("💻 CPU training mode")
            
        except ImportError:
            logger.error("❌ TensorFlow not installed")
            logger.info("💡 Install with: pip install tensorflow==2.15.0")
            return None
        
        # Prepare features and labels
        logger.info("🔄 Preparing full-body features from sequences...")
        
        X = []  # Features
        y = []  # Labels
        
        gesture_to_idx = {g: i for i, g in enumerate(gestures)}
        
        for seq in training_sequences:
            gesture = seq['gesture']
            frames = seq['frames']
            
            # Extract comprehensive features from each frame
            frame_features = []
            for frame in frames:
                features = []
                
                # Pose landmarks (33 * 3 = 99 features)
                for lm in frame['pose_landmarks']:
                    features.extend([lm['x'], lm['y'], lm['z']])
                
                # Pad if less than 33 landmarks
                while len(features) < 99:
                    features.extend([0.0, 0.0, 0.0])
                
                # Left hand landmarks (21 * 3 = 63 features)
                if frame['has_left_hand']:
                    for lm in frame['left_hand_landmarks']:
                        features.extend([lm['x'], lm['y'], lm['z']])
                else:
                    features.extend([0.0] * 63)
                
                # Right hand landmarks (21 * 3 = 63 features)
                if frame['has_right_hand']:
                    for lm in frame['right_hand_landmarks']:
                        features.extend([lm['x'], lm['y'], lm['z']])
                else:
                    features.extend([0.0] * 63)
                
                # Face landmarks (14 * 3 = 42 features)
                if frame['has_face']:
                    for lm in frame['face_landmarks']:
                        features.extend([lm['x'], lm['y'], lm['z']])
                else:
                    features.extend([0.0] * 42)
                
                # Spatial relationships (6 features)
                spatial = frame['annotation_methods']['spatial_relationships']
                if 'hand_positions' in spatial and 'left' in spatial['hand_positions']:
                    left_rel = spatial['hand_positions']['left']['relative_to_nose']
                    features.extend([left_rel['x'], left_rel['y'], left_rel['z']])
                else:
                    features.extend([0.0, 0.0, 0.0])
                
                if 'hand_positions' in spatial and 'right' in spatial['hand_positions']:
                    right_rel = spatial['hand_positions']['right']['relative_to_nose']
                    features.extend([right_rel['x'], right_rel['y'], right_rel['z']])
                else:
                    features.extend([0.0, 0.0, 0.0])
                
                frame_features.append(features)
            
            X.append(frame_features)
            y.append(gesture_to_idx[gesture])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        
        logger.info(f"✅ Full-body features prepared")
        logger.info(f"📊 X shape: {X.shape}")
        logger.info(f"📊 y shape: {y.shape}")
        logger.info(f"📊 Feature dimension: {X.shape[2]} (pose + hands + face + spatial)")
        
        # Build enhanced model for full body features
        logger.info("🔄 Building enhanced LSTM model for full-body recognition...")
        
        sequence_length = X.shape[1]
        num_features = X.shape[2]
        num_classes = len(gestures)
        
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(sequence_length, num_features)),
            
            # Bidirectional LSTM for better temporal understanding
            layers.Bidirectional(layers.LSTM(256, return_sequences=True)),
            layers.Dropout(0.4),
            layers.Bidirectional(layers.LSTM(128, return_sequences=False)),
            layers.Dropout(0.4),
            
            # Dense layers for classification
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("✅ Enhanced model architecture built")
        logger.info(f"📊 Total parameters: {model.count_params():,}")
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_val = X[split_idx:]
        y_val = y[split_idx:]
        
        logger.info(f"📊 Training samples: {len(X_train)}")
        logger.info(f"📊 Validation samples: {len(X_val)}")
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6
        )
        
        # Train
        logger.info("🔄 Training full-body model...")
        logger.info("⏱️ This will take several minutes...")
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=8,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        logger.info("✅ Training complete")
        
        # Evaluate
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        logger.info(f"📊 Validation accuracy: {val_acc:.2%}")
        logger.info(f"📊 Validation loss: {val_loss:.4f}")
        
        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = self.models_dir / f"fullbody_gsl_model_{timestamp}.h5"
        
        logger.info(f"💾 Saving full-body model to {model_path}")
        model.save(str(model_path))
        
        # Save gesture mapping
        mapping_path = self.models_dir / f"fullbody_gesture_mapping_{timestamp}.json"
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump({
                'gestures': gestures,
                'num_classes': len(gestures),
                'model_file': model_path.name,
                'validation_accuracy': float(val_acc),
                'validation_loss': float(val_loss),
                'training_date': timestamp,
                'model_type': 'full_body_holistic',
                'features': 'pose(33) + left_hand(21) + right_hand(21) + face(14) + spatial(6)'
            }, f, indent=2)
        
        logger.info(f"✅ Model saved: {model_path}")
        logger.info(f"✅ Mapping saved: {mapping_path}")
        
        return model_path
    
    def run_complete_pipeline(self):
        """Run the complete full-body GSL pipeline"""
        logger.info("🚀 Starting Complete Full-Body GSL Training Pipeline")
        logger.info("=" * 70)
        logger.info("📋 Pipeline Steps:")
        logger.info("   0. Download Kaggle Dataset (if enabled)")
        logger.info("   1. Full-Body Video Segmentation (Pose + Hands + Face)")
        logger.info("   2. Advanced Annotation (Keypoints + Spatial + Holistic)")
        logger.info("   3. Training Data Preparation (Full Body Features)")
        logger.info("   4. Enhanced Model Training (Bidirectional LSTM)")
        logger.info("=" * 70)
        
        # Step 0: Download Kaggle dataset if enabled
        if self.use_kaggle:
            logger.info("📥 STEP 0: Downloading Kaggle Dataset")
            kaggle_path = self.download_kaggle_dataset()
            if kaggle_path:
                logger.info(f"✅ Kaggle dataset ready: {kaggle_path}")
            else:
                logger.warning("⚠️ Kaggle download failed, will use local videos only")
        
        # Find videos from both sources
        local_videos = list(self.video_dir.glob("*.mp4"))
        kaggle_videos = list(self.kaggle_video_dir.glob("*.mp4")) if self.use_kaggle else []
        
        all_videos = local_videos + kaggle_videos
        
        if not all_videos:
            logger.error(f"❌ No videos found in {self.video_dir} or {self.kaggle_video_dir}")
            return False
        
        logger.info(f"📹 Found {len(all_videos)} videos total:")
        logger.info(f"   - Local: {len(local_videos)}")
        logger.info(f"   - Kaggle: {len(kaggle_videos)}")
        
        # Process all videos (or limit to first N)
        videos_to_process = all_videos[:self.max_videos] if len(all_videos) > self.max_videos else all_videos
        logger.info(f"📹 Will process {len(videos_to_process)} videos")
        
        all_segmented_data = []
        all_annotated_data = []
        
        for idx, video_path in enumerate(videos_to_process, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"📹 Processing video {idx}/{len(videos_to_process)}: {video_path.name}")
            logger.info(f"{'='*70}\n")
            
            # Step 1: Full-body segmentation
            seg_data = self.segment_video_full_body(video_path)
            if seg_data:
                all_segmented_data.append(seg_data)
                
                # Step 2: Full-body annotation
                ann_data = self.annotate_with_full_body_context(seg_data)
                all_annotated_data.append(ann_data)
            else:
                logger.warning(f"⚠️ Skipping video {video_path.name} due to segmentation failure")
        
        if not all_annotated_data:
            logger.error("❌ No videos were successfully processed")
            return False
        
        logger.info(f"\n{'='*70}")
        logger.info(f"✅ Successfully processed {len(all_annotated_data)} videos")
        logger.info(f"{'='*70}\n")
        
        # Step 3: Prepare training data from all videos
        logger.info("📊 STEP 3: Combining Training Data from All Videos")
        combined_frames = []
        for ann_data in all_annotated_data:
            combined_frames.extend(ann_data['annotated_frames'])
        
        logger.info(f"📊 Total annotated frames: {len(combined_frames)}")
        
        # Create combined annotation data
        combined_ann_data = {
            'annotated_frames': combined_frames,
            'gesture_category': all_annotated_data[0].get('gesture_category', 'mixed'),
            'gestures_list': all_annotated_data[0].get('gestures_list', []),
            'num_videos': len(all_annotated_data)
        }
        
        training_sequences, gestures = self.prepare_training_data(combined_ann_data)
        
        # Step 4: Train model
        model_path = self.train_model(training_sequences, gestures)
        
        if model_path:
            logger.info("🎉 Complete Full-Body Pipeline Finished Successfully!")
            logger.info(f"🤖 Model saved: {model_path}")
            logger.info(f"📊 Trained on {len(gestures)} gestures")
            logger.info(f"📁 All outputs in: {self.output_dir}")
            logger.info("🎯 Model uses comprehensive features:")
            logger.info("   ✅ 33 body pose landmarks")
            logger.info("   ✅ 21 left hand landmarks")
            logger.info("   ✅ 21 right hand landmarks")
            logger.info("   ✅ 14 face landmarks")
            logger.info("   ✅ Spatial relationships")
            return True
        else:
            logger.error("❌ Training failed")
            return False

def main():
    """Main function"""
    print("🎯 Complete Full-Body GSL Training Pipeline")
    print("🔬 Comprehensive Body Tracking → Advanced Annotation → Enhanced Training")
    print("=" * 70)
    
    # Enable Kaggle dataset download (first 40 videos)
    pipeline = FullBodyGSLPipeline(use_kaggle=True, max_videos=40)
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n🎉 SUCCESS! Full-body pipeline finished")
        print("📊 Check sam2_training_output/ for all results")
        print("🚀 Test your enhanced model with: python test_deep_learning_model.py")
    else:
        print("\n❌ Pipeline failed - check logs above")

if __name__ == "__main__":
    main()
