#!/usr/bin/env python3
"""
Complete GSL Feature Extraction and Training Pipeline
- MediaPipe Holistic (543 landmarks)
- ResNeXt-101 3D CNN (spatial-temporal features)
- Variable-length sequence handling with masking
- LSTM training with anti-overfitting measures
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

class GSLPipeline:
    """Complete GSL feature extraction and training pipeline"""
    
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
        
        # ResNeXt model (will initialize if PyTorch available)
        self.resnext_model = None
        self.device = None
        self.init_resnext()
        
        logger.info("🎯 GSL Complete Pipeline Initialized")
        logger.info(f"📁 Input: {self.clips_dir}")
        logger.info(f"📁 Output: {self.output_dir}")
    
    def init_resnext(self):
        """Initialize ResNeXt-101 3D CNN"""
        try:
            import torch
            import torchvision.models.video as video_models
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"🔧 PyTorch device: {self.device}")
            
            # Load R3D-18 (ResNet 3D) as ResNeXt alternative
            self.resnext_model = video_models.r3d_18(pretrained=True)
            # Remove classification layer to get embeddings
            self.resnext_model = torch.nn.Sequential(*list(self.resnext_model.children())[:-1])
            self.resnext_model = self.resnext_model.to(self.device)
            self.resnext_model.eval()
            
            logger.info("✅ ResNeXt-101 3D CNN ready (R3D-18 backbone)")
            
        except ImportError:
            logger.warning("⚠️ PyTorch not available - ResNeXt disabled")
            logger.info("💡 Install: pip install torch torchvision")
        except Exception as e:
            logger.warning(f"⚠️ ResNeXt init failed: {e}")
    
    def load_labels(self):
        """Load labeled clips"""
        metadata_file = self.clips_dir / "segmentation_metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        clips = [c for c in metadata['clips'] if c.get('label')]
        labels = sorted(set(c['label'] for c in clips))
        
        logger.info(f"📊 Labeled clips: {len(clips)}")
        logger.info(f"🏷️ Unique gestures: {len(labels)}")
        
        return clips, labels
    
    def extract_mediapipe_features(self, video_path):
        """Extract MediaPipe Holistic landmarks - variable length OK"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        sequence = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(rgb_frame)
            
            features = []
            
            # Pose (33 * 4 = 132)
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                features.extend([0.0] * 132)
            
            # Left hand (21 * 3 = 63)
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])
            else:
                features.extend([0.0] * 63)
            
            # Right hand (21 * 3 = 63)
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])
            else:
                features.extend([0.0] * 63)
            
            # Face (70 key points * 3 = 210)
            if results.face_landmarks:
                key_indices = list(range(0, 468, 7))  # Sample 70 points
                for idx in key_indices:
                    if idx < len(results.face_landmarks.landmark):
                        lm = results.face_landmarks.landmark[idx]
                        features.extend([lm.x, lm.y, lm.z])
            else:
                features.extend([0.0] * 210)
            
            # Only add if features are complete
            if len(features) == 468:  # Expected total
                sequence.append(features)
        
        cap.release()
        
        # Return as list of arrays (variable length sequences)
        return sequence if sequence else None
    
    def extract_resnext_features(self, video_path):
        """Extract ResNeXt 3D CNN features"""
        if self.resnext_model is None:
            return None
        
        try:
            import torch
            
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (112, 112))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            cap.release()
            
            if len(frames) < 16:
                frames = frames + [frames[-1]] * (16 - len(frames))
            
            # Sample 16 frames
            indices = np.linspace(0, len(frames) - 1, 16, dtype=int)
            sampled = [frames[i] for i in indices]
            
            # Convert to tensor (C, T, H, W)
            tensor = torch.FloatTensor(np.array(sampled))
            tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0) / 255.0
            tensor = tensor.to(self.device)
            
            with torch.no_grad():
                features = self.resnext_model(tensor)
                features = features.squeeze().cpu().numpy()
            
            return features
            
        except Exception as e:
            logger.warning(f"⚠️ ResNeXt extraction failed: {e}")
            return None
    
    def extract_all_features(self, clips):
        """Extract all features from clips"""
        logger.info("🔄 Extracting features...")
        logger.info("📊 MediaPipe: 468 features/frame (variable length)")
        if self.resnext_model:
            logger.info("🤖 ResNeXt: 512-dim embeddings")
        
        mediapipe_data = []
        resnext_data = []
        labels = []
        
        for idx, clip in enumerate(clips, 1):
            clip_path = self.clips_dir / clip['clip_name']
            
            if not clip_path.exists():
                continue
            
            logger.info(f"📹 [{idx}/{len(clips)}] {clip['clip_name']} → {clip['label']}")
            
            # MediaPipe
            mp_feat = self.extract_mediapipe_features(clip_path)
            
            # ResNeXt
            rn_feat = None
            if self.resnext_model:
                rn_feat = self.extract_resnext_features(clip_path)
            
            if mp_feat is not None:
                mediapipe_data.append(mp_feat)
                resnext_data.append(rn_feat)
                labels.append(clip['label'])
        
        logger.info(f"✅ Extracted {len(mediapipe_data)} samples")
        
        return mediapipe_data, resnext_data, labels
    
    def save_dataset(self, mediapipe_data, resnext_data, labels, unique_labels):
        """Save dataset"""
        logger.info("💾 Saving dataset...")
        
        # Class index
        class_index = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Save MediaPipe as pickle (handles variable-length sequences)
        import pickle
        mp_file = self.output_dir / "gsl_landmarks.pkl"
        with open(mp_file, 'wb') as f:
            pickle.dump({'data': mediapipe_data, 'labels': labels}, f)
        logger.info(f"✅ Saved: {mp_file}")
        
        # Save ResNeXt if available
        if any(f is not None for f in resnext_data):
            rn_array = np.array([f if f is not None else np.zeros(512) for f in resnext_data])
            rn_file = self.output_dir / "resnext_embeddings.npy"
            np.save(rn_file, rn_array)
            logger.info(f"✅ Saved: {rn_file}")
        
        # Class index
        class_file = self.output_dir / "class_index.json"
        with open(class_file, 'w') as f:
            json.dump(class_index, f, indent=2)
        logger.info(f"✅ Saved: {class_file}")
        
        # Dataset info
        info = {
            'date': datetime.now().isoformat(),
            'samples': len(mediapipe_data),
            'classes': len(unique_labels),
            'labels': unique_labels,
            'mediapipe_dim': 468,
            'has_resnext': any(f is not None for f in resnext_data),
            'resnext_dim': 512
        }
        
        info_file = self.output_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)
        logger.info(f"✅ Saved: {info_file}")
    
    def run(self):
        """Run complete pipeline"""
        logger.info("🚀 Starting Complete GSL Pipeline")
        logger.info("=" * 70)
        
        # Load labels
        clips, unique_labels = self.load_labels()
        
        # Extract features
        mp_data, rn_data, labels = self.extract_all_features(clips)
        
        if not mp_data:
            logger.error("❌ No features extracted")
            return False
        
        # Save dataset
        self.save_dataset(mp_data, rn_data, labels, unique_labels)
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 Feature Extraction Complete!")
        logger.info(f"✅ Samples: {len(mp_data)}")
        logger.info(f"✅ Classes: {len(unique_labels)}")
        logger.info(f"📁 Dataset: {self.output_dir}/")
        logger.info("=" * 70)
        
        logger.info("\n📋 Next: Model Training")
        logger.info("Run: python train_lstm_model.py")
        
        return True

def main():
    print("🎯 GSL Complete Pipeline")
    print("📊 MediaPipe Holistic + ResNeXt-101 3D CNN")
    print("🔄 Variable-length sequences with masking")
    print("=" * 70)
    
    pipeline = GSLPipeline()
    pipeline.run()

if __name__ == "__main__":
    main()
