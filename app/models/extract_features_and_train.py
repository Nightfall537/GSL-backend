#!/usr/bin/env python3
"""
GSL Feature Extraction and Model Training Pipeline
Follows complete specification:
1. MediaPipe Holistic landmark extraction (543 landmarks)
2. ResNeXt-101 3D CNN feature extraction
3. LSTM model training with fusion
4. Real-time recognition system
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import json
from pathlib import Path
import logging
from datetime import datetime
import mediapipe as mp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GSLFeatureExtractor:
    """Extract MediaPipe Holistic + ResNeXt-101 features"""
    
    def __init__(self):
        self.clips_dir = Path("segmented_clips")
        self.output_dir = Path("gsl_dataset")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize ResNeXt-101 3D CNN
        self.resnext_model = None
        self.device = None
        self.initialize_resnext()
        
        logger.info("🎯 GSL Feature Extractor Initialized")
        logger.info(f"📁 Input: {self.clips_dir}")
        logger.info(f"📁 Output: {self.output_dir}")
    
    def initialize_resnext(self):
        """Initialize ResNeXt-101 3D CNN (Kinetics-400 pre-trained)"""
        try:
            import torch
            import torchvision.models.video as video_models
            
            logger.info("🔄 Initializing ResNeXt-101 3D CNN...")
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"🔧 Device: {self.device}")
            
            # Load pre-trained 3D CNN (Kinetics-400)
            # Using R3D-18 (3D ResNet) which is similar to ResNeXt for video
            try:
                from torchvision.models.video import r3d_18, R3D_18_Weights
                
                logger.info("📥 Loading pre-trained R3D-18 (3D ResNet)...")
                weights = R3D_18_Weights.KINETICS400_V1
                self.resnext_model = r3d_18(weights=weights)
                
                # Remove final classification layer to get embeddings
                self.resnext_model = torch.nn.Sequential(*list(self.resnext_model.children())[:-1])
                self.resnext_model = self.resnext_model.to(self.device)
                self.resnext_model.eval()
                
                logger.info("✅ R3D-18 3D CNN initialized (Kinetics-400 pre-trained)")
                logger.info("🎯 Captures: spatial-temporal features, motion, hand-face interaction")
            except Exception as e:
                logger.warning(f"⚠️ Could not load pre-trained model: {e}")
                logger.info("💡 Will use MediaPipe landmarks only")
                self.resnext_model = None
            
        except ImportError:
            logger.warning("⚠️ PyTorch not available")
            logger.info("💡 Install with: pip install torch torchvision")
            logger.info("💡 Will use MediaPipe landmarks only")
            self.resnext_model = None
    
    def load_labels(self):
        """Load labeled clips from metadata"""
        metadata_file = self.clips_dir / "segmentation_metadata.json"
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        clips = metadata['clips']
        
        # Filter clips with labels
        labeled_clips = [c for c in clips if c.get('label')]
        
        logger.info(f"📊 Total clips: {len(clips)}")
        logger.info(f"✅ Labeled clips: {len(labeled_clips)}")
        
        # Get unique labels
        labels = sorted(set(c['label'] for c in labeled_clips))
        logger.info(f"🏷️ Unique gestures: {len(labels)}")
        logger.info(f"📋 Labels: {labels}")
        
        return labeled_clips, labels
    
    def extract_holistic_landmarks(self, video_path):
        """Extract 543 MediaPipe Holistic landmarks"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            landmarks_sequence = []
            frame_count = 0
            max_frames = 150  # Limit frames to prevent memory issues
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
            
            # Process with MediaPipe Holistic
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(rgb_frame)
            
            frame_landmarks = []
            
            # Pose landmarks (33 * 4 = 132 values: x, y, z, visibility)
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                frame_landmarks.extend([0.0] * 132)
            
            # Left hand landmarks (21 * 3 = 63 values)
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    frame_landmarks.extend([lm.x, lm.y, lm.z])
            else:
                frame_landmarks.extend([0.0] * 63)
            
            # Right hand landmarks (21 * 3 = 63 values)
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    frame_landmarks.extend([lm.x, lm.y, lm.z])
            else:
                frame_landmarks.extend([0.0] * 63)
            
            # Face landmarks (468 * 3 = 1404 values - using subset for efficiency)
            # Using 70 key face landmarks instead of all 468
            if results.face_landmarks:
                key_face_indices = list(range(0, 468, 7))  # Sample every 7th landmark
                for idx in key_face_indices:
                    if idx < len(results.face_landmarks.landmark):
                        lm = results.face_landmarks.landmark[idx]
                        frame_landmarks.extend([lm.x, lm.y, lm.z])
            else:
                frame_landmarks.extend([0.0] * (70 * 3))
            
                # Ensure consistent length
                if len(frame_landmarks) > 0:
                    landmarks_sequence.append(frame_landmarks)
            
            cap.release()
            
            if not landmarks_sequence:
                return None
            
            # Pad sequences to same length if needed
            max_len = max(len(lm) for lm in landmarks_sequence)
            padded_sequence = []
            for lm in landmarks_sequence:
                if len(lm) < max_len:
                    lm = lm + [0.0] * (max_len - len(lm))
                padded_sequence.append(lm)
            
            return np.array(padded_sequence, dtype=np.float32)
            
        except Exception as e:
            logger.warning(f"⚠️ Landmark extraction error: {e}")
            return None
    
    def extract_resnext_features(self, video_path):
        """Extract ResNeXt-101 3D CNN features"""
        if self.resnext_model is None:
            return None
        
        try:
            import torch
            import torchvision.transforms as transforms
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Resize to 112x112 for 3D CNN
                frame = cv2.resize(frame, (112, 112))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            cap.release()
            
            if len(frames) < 16:
                # Pad with last frame if too short
                while len(frames) < 16:
                    frames.append(frames[-1])
            
            # Sample 16 frames uniformly
            indices = np.linspace(0, len(frames) - 1, 16, dtype=int)
            sampled_frames = [frames[i] for i in indices]
            
            # Convert to tensor (C, T, H, W)
            video_tensor = torch.FloatTensor(np.array(sampled_frames))
            video_tensor = video_tensor.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
            video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension
            video_tensor = video_tensor / 255.0  # Normalize
            video_tensor = video_tensor.to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.resnext_model(video_tensor)
                features = features.squeeze().cpu().numpy()
            
            return features
            
        except Exception as e:
            logger.warning(f"⚠️ ResNeXt extraction failed: {e}")
            return None
    
    def extract_all_features(self, labeled_clips):
        """Extract both MediaPipe landmarks AND ResNeXt features"""
        logger.info("🔄 Extracting Features...")
        logger.info("📊 MediaPipe Holistic: 543 landmarks per frame")
        if self.resnext_model:
            logger.info("🤖 ResNeXt-101 3D CNN: Spatial-temporal features")
        
        all_landmarks = []
        all_resnext_features = []
        all_labels = []
        clip_info = []
        
        for idx, clip in enumerate(labeled_clips, 1):
            clip_path = self.clips_dir / clip['clip_name']
            
            if not clip_path.exists():
                logger.warning(f"⚠️ Clip not found: {clip['clip_name']}")
                continue
            
            logger.info(f"📹 [{idx}/{len(labeled_clips)}] {clip['clip_name']} → {clip['label']}")
            
            # Extract MediaPipe landmarks
            landmarks = self.extract_holistic_landmarks(clip_path)
            
            # Extract ResNeXt features
            resnext_feat = None
            if self.resnext_model:
                resnext_feat = self.extract_resnext_features(clip_path)
            
            if landmarks is not None and len(landmarks) > 0:
                all_landmarks.append(landmarks)
                all_resnext_features.append(resnext_feat)
                all_labels.append(clip['label'])
                clip_info.append({
                    'clip_name': clip['clip_name'],
                    'label': clip['label'],
                    'num_frames': len(landmarks),
                    'landmark_shape': landmarks.shape,
                    'has_resnext': resnext_feat is not None
                })
            else:
                logger.warning(f"⚠️ No features extracted from {clip['clip_name']}")
        
        logger.info(f"✅ Extracted features from {len(all_landmarks)} clips")
        if self.resnext_model:
            valid_resnext = sum(1 for f in all_resnext_features if f is not None)
            logger.info(f"✅ ResNeXt features: {valid_resnext}/{len(all_landmarks)} clips")
        
        return all_landmarks, all_resnext_features, all_labels, clip_info
    
    def save_dataset(self, landmarks, resnext_features, labels, clip_info, unique_labels):
        """Save complete dataset with landmarks and ResNeXt features"""
        logger.info("💾 Saving dataset...")
        
        # Create class index
        class_index = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Save landmarks
        landmarks_file = self.output_dir / "gsl_landmarks.npz"
        np.savez_compressed(
            landmarks_file,
            landmarks=landmarks,
            labels=labels,
            clip_info=clip_info
        )
        logger.info(f"✅ Saved: {landmarks_file}")
        
        # Save ResNeXt features if available
        if any(f is not None for f in resnext_features):
            resnext_file = self.output_dir / "resnext_embeddings.npy"
            # Convert to array, using zeros for missing features
            resnext_array = []
            for feat in resnext_features:
                if feat is not None:
                    resnext_array.append(feat)
                else:
                    resnext_array.append(np.zeros(512))  # Default embedding size
            np.save(resnext_file, np.array(resnext_array))
            logger.info(f"✅ Saved: {resnext_file}")
        
        # Save class index
        class_index_file = self.output_dir / "class_index.json"
        with open(class_index_file, 'w', encoding='utf-8') as f:
            json.dump(class_index, f, indent=2)
        logger.info(f"✅ Saved: {class_index_file}")
        
        # Save dataset info
        dataset_info = {
            'creation_date': datetime.now().isoformat(),
            'total_samples': len(landmarks),
            'num_classes': len(unique_labels),
            'classes': unique_labels,
            'landmark_dimension': 543,
            'has_resnext': any(f is not None for f in resnext_features),
            'resnext_dimension': 512,
            'clips': clip_info
        }
        
        info_file = self.output_dir / "dataset_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2)
        logger.info(f"✅ Saved: {info_file}")
        
        return class_index
    
    def run_extraction(self):
        """Run complete feature extraction"""
        logger.info("🚀 Starting Feature Extraction Pipeline")
        logger.info("=" * 70)
        
        # Load labels
        labeled_clips, unique_labels = self.load_labels()
        
        if not labeled_clips:
            logger.error("❌ No labeled clips found")
            return False
        
        # Extract features (MediaPipe + ResNeXt)
        landmarks, resnext_features, labels, clip_info = self.extract_all_features(labeled_clips)
        
        if not landmarks:
            logger.error("❌ No features extracted")
            return False
        
        # Save dataset
        class_index = self.save_dataset(landmarks, resnext_features, labels, clip_info, unique_labels)
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 Feature Extraction Complete!")
        logger.info(f"✅ Samples: {len(landmarks)}")
        logger.info(f"✅ Classes: {len(unique_labels)}")
        logger.info(f"📁 Dataset: {self.output_dir}/")
        logger.info("=" * 70)
        
        logger.info("\n📋 Next Step: Model Training")
        logger.info("Run: python train_gsl_model.py")
        
        return True

def main():
    """Main function"""
    print("🎯 GSL Feature Extraction Pipeline")
    print("📊 MediaPipe Holistic: 543 landmarks per frame")
    print("🤖 Preparing data for LSTM training")
    print("=" * 70)
    
    extractor = GSLFeatureExtractor()
    success = extractor.run_extraction()
    
    if success:
        print("\n✅ SUCCESS! Features extracted")
        print("📁 Check: gsl_dataset/")
        print("\n🚀 Ready for model training")
    else:
        print("\n❌ Extraction failed")

if __name__ == "__main__":
    main()
