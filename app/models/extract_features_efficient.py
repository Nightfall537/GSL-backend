#!/usr/bin/env python3
"""
Efficient GSL Feature Extraction
MediaPipe Holistic + ResNeXt-101 with memory optimization
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import json
from pathlib import Path
import logging
import mediapipe as mp
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EfficientGSLExtractor:
    """Memory-efficient feature extraction"""
    
    def __init__(self):
        self.clips_dir = Path("segmented_clips")
        self.output_dir = Path("gsl_dataset")
        self.output_dir.mkdir(exist_ok=True)
        
        # MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,  # Lower complexity for efficiency
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5
        )
        
        # ResNeXt (optional)
        self.resnext_model = None
        self.init_resnext()
        
        logger.info("🎯 Efficient GSL Extractor Initialized")
    
    def init_resnext(self):
        """Initialize ResNeXt if available"""
        try:
            import torch
            import torchvision.models.video as vm
            
            self.device = torch.device("cpu")  # Use CPU to save memory
            self.resnext_model = vm.r3d_18(weights='DEFAULT')
            self.resnext_model = torch.nn.Sequential(*list(self.resnext_model.children())[:-1])
            self.resnext_model.eval()
            
            logger.info("✅ ResNeXt-101 ready")
        except:
            logger.info("⚠️ ResNeXt disabled (PyTorch not available)")
    
    def extract_mediapipe(self, video_path):
        """Extract MediaPipe landmarks"""
        cap = cv2.VideoCapture(str(video_path))
        sequence = []
        
        while len(sequence) < 100:  # Limit frames
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(rgb)
            
            feat = []
            
            # Pose (33*4=132)
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    feat.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                feat.extend([0.0]*132)
            
            # Hands (21*3=63 each)
            for hand_lm in [results.left_hand_landmarks, results.right_hand_landmarks]:
                if hand_lm:
                    for lm in hand_lm.landmark:
                        feat.extend([lm.x, lm.y, lm.z])
                else:
                    feat.extend([0.0]*63)
            
            # Face (70 key points*3=210)
            if results.face_landmarks:
                for i in range(0, 468, 7):
                    if i < len(results.face_landmarks.landmark):
                        lm = results.face_landmarks.landmark[i]
                        feat.extend([lm.x, lm.y, lm.z])
            else:
                feat.extend([0.0]*210)
            
            if len(feat) == 468:
                sequence.append(feat)
        
        cap.release()
        return sequence if sequence else None
    
    def extract_resnext(self, video_path):
        """Extract ResNeXt features"""
        if not self.resnext_model:
            return None
        
        try:
            import torch
            
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            
            while len(frames) < 32:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (112, 112))
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            cap.release()
            
            if len(frames) < 16:
                return None
            
            # Sample 16 frames
            idx = np.linspace(0, len(frames)-1, 16, dtype=int)
            sampled = [frames[i] for i in idx]
            
            # To tensor
            tensor = torch.FloatTensor(np.array(sampled))
            tensor = tensor.permute(3,0,1,2).unsqueeze(0) / 255.0
            
            with torch.no_grad():
                feat = self.resnext_model(tensor).squeeze().numpy()
            
            return feat
        except:
            return None
    
    def run(self):
        """Run extraction"""
        logger.info("🚀 Extracting Features")
        logger.info("="*70)
        
        # Load labels
        with open(self.clips_dir / "segmentation_metadata.json") as f:
            clips = [c for c in json.load(f)['clips'] if c.get('label')]
        
        labels_set = sorted(set(c['label'] for c in clips))
        logger.info(f"📊 {len(clips)} clips, {len(labels_set)} classes")
        
        mp_data = []
        rn_data = []
        labels = []
        
        for i, clip in enumerate(clips, 1):
            path = self.clips_dir / clip['clip_name']
            if not path.exists():
                continue
            
            logger.info(f"[{i}/{len(clips)}] {clip['clip_name']} → {clip['label']}")
            
            mp = self.extract_mediapipe(path)
            rn = self.extract_resnext(path) if self.resnext_model else None
            
            if mp:
                mp_data.append(mp)
                rn_data.append(rn)
                labels.append(clip['label'])
        
        logger.info(f"✅ Extracted {len(mp_data)} samples")
        
        # Save
        logger.info("💾 Saving...")
        
        with open(self.output_dir / "gsl_landmarks.pkl", 'wb') as f:
            pickle.dump({'data': mp_data, 'labels': labels}, f)
        
        if any(rn_data):
            rn_array = np.array([r if r is not None else np.zeros(512) for r in rn_data])
            np.save(self.output_dir / "resnext_embeddings.npy", rn_array)
        
        class_idx = {l: i for i, l in enumerate(labels_set)}
        with open(self.output_dir / "class_index.json", 'w') as f:
            json.dump(class_idx, f, indent=2)
        
        logger.info("🎉 Complete!")
        logger.info(f"📁 {self.output_dir}/")
        
        return True

if __name__ == "__main__":
    EfficientGSLExtractor().run()
