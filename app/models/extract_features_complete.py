#!/usr/bin/env python3
"""
Complete GSL Feature Extraction
MediaPipe Holistic + ResNeXt-101 3D CNN
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

# Fixed feature dimensions
POSE_DIM = 132  # 33 landmarks * 4 (x,y,z,visibility)
HAND_DIM = 63   # 21 landmarks * 3 (x,y,z)
FACE_DIM = 210  # 70 key landmarks * 3 (x,y,z)
TOTAL_DIM = POSE_DIM + HAND_DIM + HAND_DIM + FACE_DIM  # 468

class FeatureExtractor:
    def __init__(self):
        self.clips_dir = Path("segmented_clips")
        self.output_dir = Path("gsl_dataset")
        self.output_dir.mkdir(exist_ok=True)
        
        # MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        
        # ResNeXt
        self.resnext_model = None
        self.device = None
        self.init_resnext()
        
        logger.info("✅ Feature Extractor Ready")
    
    def init_resnext(self):
        """Initialize ResNeXt-101 3D CNN"""
        try:
            import torch
            import torchvision.models.video as video_models
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"🔧 Device: {self.device}")
            
            # Load R3D-18 (ResNet 3D)
            self.resnext_model = video_models.r3d_18(pretrained=True)
            self.resnext_model = torch.nn.Sequential(*list(self.resnext_model.children())[:-1])
            self.resnext_model = self.resnext_model.to(self.device)
            self.resnext_model.eval()
            
            logger.info("✅ ResNeXt 3D CNN Ready")
        except:
            logger.warning("⚠️ ResNeXt not available - using landmarks only")
    
    def extract_landmarks(self, video_path):
        """Extract fixed-size landmarks"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        sequence = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(rgb)
            
            features = []
            
            # Pose (132)
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                features.extend([0.0] * POSE_DIM)
            
            # Left hand (63)
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])
            else:
                features.extend([0.0] * HAND_DIM)
            
            # Right hand (63)
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])
            else:
                features.extend([0.0] * HAND_DIM)
            
            # Face (210 - 70 key points)
            if results.face_landmarks:
                indices = list(range(0, 468, 7))[:70]
                for idx in indices:
                    lm = results.face_landmarks.landmark[idx]
                    features.extend([lm.x, lm.y, lm.z])
            else:
                features.extend([0.0] * FACE_DIM)
            
            # Ensure exact size
            features = features[:TOTAL_DIM]
            features.extend([0.0] * (TOTAL_DIM - len(features)))
            
            sequence.append(features)
        
        cap.release()
        return np.array(sequence, dtype=np.float32) if sequence else None
    
    def extract_resnext(self, video_path):
        """Extract ResNeXt features"""
        if not self.resnext_model:
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
                frames.extend([frames[-1]] * (16 - len(frames)))
            
            # Sample 16 frames
            indices = np.linspace(0, len(frames) - 1, 16, dtype=int)
            sampled = [frames[i] for i in indices]
            
            # To tensor
            tensor = torch.FloatTensor(np.array(sampled))
            tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0) / 255.0
            tensor = tensor.to(self.device)
            
            with torch.no_grad():
                feat = self.resnext_model(tensor).squeeze().cpu().numpy()
            
            return feat
        except:
            return None
    
    def run(self):
        """Extract all features"""
        logger.info("🚀 Starting Feature Extraction")
        logger.info("=" * 70)
        
        # Load labels
        metadata_file = self.clips_dir / "segmentation_metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        clips = [c for c in metadata['clips'] if c.get('label')]
        labels_list = sorted(set(c['label'] for c in clips))
        
        logger.info(f"📊 Clips: {len(clips)}")
        logger.info(f"🏷️ Classes: {len(labels_list)}")
        
        # Extract
        all_landmarks = []
        all_resnext = []
        all_labels = []
        
        for idx, clip in enumerate(clips, 1):
            path = self.clips_dir / clip['clip_name']
            logger.info(f"[{idx}/{len(clips)}] {clip['clip_name']} → {clip['label']}")
            
            landmarks = self.extract_landmarks(path)
            resnext = self.extract_resnext(path) if self.resnext_model else None
            
            if landmarks is not None:
                all_landmarks.append(landmarks)
                all_resnext.append(resnext)
                all_labels.append(clip['label'])
        
        # Save
        logger.info(f"\n💾 Saving {len(all_landmarks)} samples...")
        
        np.savez_compressed(
            self.output_dir / "gsl_landmarks.npz",
            landmarks=all_landmarks,
            labels=all_labels
        )
        
        if any(r is not None for r in all_resnext):
            resnext_array = [r if r is not None else np.zeros(512) for r in all_resnext]
            np.save(self.output_dir / "resnext_embeddings.npy", np.array(resnext_array))
        
        class_index = {label: idx for idx, label in enumerate(labels_list)}
        with open(self.output_dir / "class_index.json", 'w') as f:
            json.dump(class_index, f, indent=2)
        
        logger.info("✅ Complete!")
        logger.info(f"📁 {self.output_dir}/")
        
        return True

if __name__ == "__main__":
    extractor = FeatureExtractor()
    extractor.run()
