"""
MediaPipe Holistic Feature Extractor
Extracts 468 landmarks per frame from video clips
"""
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Dict, Optional, List
import json

from ..utils.logger import setup_logger
from ..config.training_config import FEATURE_CONFIG, SEGMENTED_CLIPS_DIR, METADATA_FILE, OUTPUT_DIR

logger = setup_logger(__name__)


class MediaPipeFeatureExtractor:
    """Extract MediaPipe Holistic landmarks from video clips"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize MediaPipe Holistic
        
        Args:
            config: Configuration dict (uses FEATURE_CONFIG if None)
        """
        self.config = config or FEATURE_CONFIG['mediapipe']
        
        # Initialize MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=self.config['static_image_mode'],
            model_complexity=self.config['model_complexity'],
            enable_segmentation=self.config['enable_segmentation'],
            refine_face_landmarks=self.config['refine_face_landmarks'],
            min_detection_confidence=self.config['min_detection_confidence'],
            min_tracking_confidence=self.config['min_tracking_confidence']
        )
        
        # Feature dimensions
        self.POSE_DIM = 132  # 33 landmarks × 4 (x, y, z, visibility)
        self.HAND_DIM = 63   # 21 landmarks × 3 (x, y, z)
        self.FACE_DIM = 210  # 70 key landmarks × 3 (x, y, z) - sampled from 468
        self.TOTAL_DIM = self.POSE_DIM + self.HAND_DIM + self.HAND_DIM + self.FACE_DIM  # 468
        
        logger.info("✅ MediaPipe Holistic initialized")
        logger.info(f"   Model complexity: {self.config['model_complexity']}")
        logger.info(f"   Feature dimension: {self.TOTAL_DIM}")
    
    def extract_landmarks(self, video_path: Path) -> Optional[np.ndarray]:
        """
        Extract 468 landmarks per frame from video
        
        Args:
            video_path: Path to video file
        
        Returns:
            np.ndarray: Shape [num_frames, 468] or None if extraction fails
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.warning(f"Cannot open video: {video_path.name}")
                return None
            
            landmarks_sequence = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe Holistic
                results = self.holistic.process(rgb_frame)
                
                # Extract frame landmarks
                frame_landmarks = self._extract_frame_landmarks(results)
                
                # Validate dimensions
                if len(frame_landmarks) == self.TOTAL_DIM:
                    landmarks_sequence.append(frame_landmarks)
                    frame_count += 1
                else:
                    logger.warning(f"Frame {frame_count} has {len(frame_landmarks)} features, expected {self.TOTAL_DIM}")
            
            cap.release()
            
            if not landmarks_sequence:
                logger.warning(f"No landmarks extracted from {video_path.name}")
                return None
            
            return np.array(landmarks_sequence, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting landmarks from {video_path.name}: {e}")
            return None
    
    def _extract_frame_landmarks(self, results) -> List[float]:
        """
        Extract landmarks from a single frame
        
        Args:
            results: MediaPipe Holistic results
        
        Returns:
            List[float]: Flattened landmarks (468 values)
        """
        features = []
        
        # Pose landmarks (33 × 4 = 132)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            features.extend([0.0] * self.POSE_DIM)
        
        # Left hand landmarks (21 × 3 = 63)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
        else:
            features.extend([0.0] * self.HAND_DIM)
        
        # Right hand landmarks (21 × 3 = 63)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
        else:
            features.extend([0.0] * self.HAND_DIM)
        
        # Face landmarks (70 key points × 3 = 210)
        # Sample every 7th landmark from 468 face landmarks
        if results.face_landmarks:
            key_face_indices = list(range(0, 468, 7))[:70]  # Get exactly 70 landmarks
            for idx in key_face_indices:
                if idx < len(results.face_landmarks.landmark):
                    lm = results.face_landmarks.landmark[idx]
                    features.extend([lm.x, lm.y, lm.z])
                else:
                    features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([0.0] * self.FACE_DIM)
        
        # Ensure exact size
        if len(features) < self.TOTAL_DIM:
            features.extend([0.0] * (self.TOTAL_DIM - len(features)))
        elif len(features) > self.TOTAL_DIM:
            features = features[:self.TOTAL_DIM]
        
        return features
    
    def validate_feature_dimensions(self, features: np.ndarray) -> bool:
        """
        Validate that features have correct dimensions
        
        Args:
            features: Feature array [num_frames, feature_dim]
        
        Returns:
            bool: True if valid, False otherwise
        """
        if features is None:
            return False
        
        if len(features.shape) != 2:
            logger.error(f"Invalid shape: {features.shape}, expected 2D array")
            return False
        
        if features.shape[1] != self.TOTAL_DIM:
            logger.error(f"Invalid feature dimension: {features.shape[1]}, expected {self.TOTAL_DIM}")
            return False
        
        if np.isnan(features).any():
            logger.error("Features contain NaN values")
            return False
        
        if np.isinf(features).any():
            logger.error("Features contain infinite values")
            return False
        
        return True
    
    def load_labeled_clips(self) -> tuple:
        """
        Load labeled clips from metadata file
        
        Returns:
            tuple: (labeled_clips, unique_labels)
        """
        try:
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            clips = metadata['clips']
            labeled_clips = [c for c in clips if c.get('label')]
            
            unique_labels = sorted(set(c['label'] for c in labeled_clips))
            
            logger.info(f"📊 Total clips: {len(clips)}")
            logger.info(f"✅ Labeled clips: {len(labeled_clips)}")
            logger.info(f"🏷️  Unique gestures: {len(unique_labels)}")
            
            return labeled_clips, unique_labels
            
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise
    
    def extract_all_features(self) -> tuple:
        """
        Extract features from all labeled clips
        
        Returns:
            tuple: (all_landmarks, all_labels, clip_info)
        """
        labeled_clips, unique_labels = self.load_labeled_clips()
        
        all_landmarks = []
        all_labels = []
        clip_info = []
        
        logger.info("🔄 Extracting MediaPipe landmarks from all clips...")
        
        for idx, clip in enumerate(labeled_clips, 1):
            clip_path = SEGMENTED_CLIPS_DIR / clip['clip_name']
            
            if not clip_path.exists():
                logger.warning(f"⚠️  Clip not found: {clip['clip_name']}")
                continue
            
            logger.info(f"📹 [{idx}/{len(labeled_clips)}] {clip['clip_name']} → {clip['label']}")
            
            # Extract landmarks
            landmarks = self.extract_landmarks(clip_path)
            
            if landmarks is not None and self.validate_feature_dimensions(landmarks):
                all_landmarks.append(landmarks)
                all_labels.append(clip['label'])
                clip_info.append({
                    'clip_name': clip['clip_name'],
                    'label': clip['label'],
                    'num_frames': len(landmarks),
                    'landmark_shape': landmarks.shape,
                    'duration': clip.get('duration', 0)
                })
                logger.info(f"   ✅ Extracted {len(landmarks)} frames")
            else:
                logger.warning(f"   ⚠️  Failed to extract valid features")
        
        logger.info(f"\n✅ Successfully extracted features from {len(all_landmarks)}/{len(labeled_clips)} clips")
        
        return all_landmarks, all_labels, clip_info, unique_labels
    
    def save_to_json(self, landmarks: List[np.ndarray], labels: List[str], 
                     clip_info: List[Dict], unique_labels: List[str]):
        """
        Save landmarks to JSON file
        
        Args:
            landmarks: List of landmark arrays
            labels: List of labels
            clip_info: List of clip metadata
            unique_labels: List of unique gesture labels
        """
        from datetime import datetime
        
        output_file = OUTPUT_DIR / "gsl_landmarks.json"
        
        # Convert numpy arrays to lists for JSON serialization
        clips_data = []
        for lm, label, info in zip(landmarks, labels, clip_info):
            clips_data.append({
                'clip_name': info['clip_name'],
                'label': label,
                'landmarks': lm.tolist(),  # Convert to list
                'num_frames': info['num_frames'],
                'duration': info['duration']
            })
        
        data = {
            'clips': clips_data,
            'metadata': {
                'total_clips': len(landmarks),
                'landmark_dimension': self.TOTAL_DIM,
                'num_classes': len(unique_labels),
                'classes': unique_labels,
                'extraction_date': datetime.now().isoformat()
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"💾 Saved landmarks to: {output_file}")
        logger.info(f"   Total clips: {len(landmarks)}")
        logger.info(f"   Feature dimension: {self.TOTAL_DIM}")
    
    def __del__(self):
        """Cleanup MediaPipe resources"""
        if hasattr(self, 'holistic'):
            self.holistic.close()
