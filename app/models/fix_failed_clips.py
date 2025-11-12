#!/usr/bin/env python3
"""
Fix MediaPipe extraction for failed clips
Disables segmentation to avoid resolution mismatch errors
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
from gsl_training_pipeline.utils.logger import setup_logger

logger = setup_logger(__name__)

# Failed clips from v6 and v7 series
FAILED_CLIPS = [
    'v6_clip_001.mp4', 'v6_clip_002.mp4', 'v6_clip_003.mp4', 'v6_clip_004.mp4',
    'v6_clip_005.mp4', 'v6_clip_006.mp4', 'v6_clip_007.mp4', 'v6_clip_008.mp4',
    'v6_clip_009.mp4', 'v7_clip_001.mp4', 'v7_clip_002.mp4', 'v7_clip_003.mp4',
    'v7_clip_004.mp4', 'v7_clip_005.mp4', 'v7_clip_006.mp4', 'v7_clip_007.mp4',
    'v7_clip_008.mp4', 'v7_clip_009.mp4', 'v7_clip_010.mp4', 'v7_clip_011.mp4',
    'v7_clip_012.mp4', 'v7_clip_013.mp4'
]

def extract_landmarks_no_segmentation(video_path: Path):
    """
    Extract landmarks WITHOUT segmentation to avoid resolution errors
    """
    # Initialize MediaPipe Holistic WITHOUT segmentation
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,  # DISABLED to fix errors
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
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
            results = holistic.process(rgb_frame)
            
            # Extract frame landmarks
            features = []
            
            # Pose landmarks (33 × 4 = 132)
            if results.pose_landmarks:
                for lm in results.pose_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                features.extend([0.0] * 132)
            
            # Left hand landmarks (21 × 3 = 63)
            if results.left_hand_landmarks:
                for lm in results.left_hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])
            else:
                features.extend([0.0] * 63)
            
            # Right hand landmarks (21 × 3 = 63)
            if results.right_hand_landmarks:
                for lm in results.right_hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])
            else:
                features.extend([0.0] * 63)
            
            # Face landmarks (70 key points × 3 = 210)
            if results.face_landmarks:
                key_face_indices = list(range(0, 468, 7))[:70]
                for idx in key_face_indices:
                    if idx < len(results.face_landmarks.landmark):
                        lm = results.face_landmarks.landmark[idx]
                        features.extend([lm.x, lm.y, lm.z])
                    else:
                        features.extend([0.0, 0.0, 0.0])
            else:
                features.extend([0.0] * 210)
            
            # Ensure exact size (468)
            if len(features) < 468:
                features.extend([0.0] * (468 - len(features)))
            elif len(features) > 468:
                features = features[:468]
            
            landmarks_sequence.append(features)
            frame_count += 1
        
        cap.release()
        holistic.close()
        
        if not landmarks_sequence:
            logger.warning(f"No landmarks extracted from {video_path.name}")
            return None
        
        return np.array(landmarks_sequence, dtype=np.float32)
        
    except Exception as e:
        logger.error(f"Error extracting landmarks from {video_path.name}: {e}")
        holistic.close()
        return None


def main():
    """Fix failed clips and update landmarks file"""
    logger.info("=" * 80)
    logger.info("🔧 Fixing MediaPipe extraction for 22 failed clips")
    logger.info("=" * 80)
    logger.info("💡 Disabling segmentation to avoid resolution mismatch errors")
    
    clips_dir = Path("segmented_clips")
    output_dir = Path("gsl_dataset")
    metadata_file = clips_dir / "segmentation_metadata.json"
    landmarks_file = output_dir / "gsl_landmarks.json"
    
    # Load existing landmarks
    with open(landmarks_file, 'r', encoding='utf-8') as f:
        landmarks_data = json.load(f)
    
    # Load metadata
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Get clip info for failed clips
    all_clips = {c['clip_name']: c for c in metadata['clips']}
    
    # Extract landmarks for failed clips
    fixed_count = 0
    for idx, clip_name in enumerate(FAILED_CLIPS, 1):
        if clip_name not in all_clips:
            logger.warning(f"Clip not in metadata: {clip_name}")
            continue
        
        clip_info = all_clips[clip_name]
        clip_path = clips_dir / clip_name
        
        if not clip_path.exists():
            logger.warning(f"Clip file not found: {clip_name}")
            continue
        
        logger.info(f"🔧 [{idx}/{len(FAILED_CLIPS)}] {clip_name} → {clip_info['label']}")
        
        # Extract landmarks
        landmarks = extract_landmarks_no_segmentation(clip_path)
        
        if landmarks is not None and landmarks.shape[1] == 468:
            # Add to landmarks data
            landmarks_data['clips'].append({
                'clip_name': clip_name,
                'label': clip_info['label'],
                'landmarks': landmarks.tolist(),
                'num_frames': len(landmarks),
                'duration': clip_info.get('duration', 0)
            })
            fixed_count += 1
            logger.info(f"   ✅ Extracted {len(landmarks)} frames")
        else:
            logger.warning(f"   ⚠️  Failed to extract valid features")
    
    # Update metadata
    landmarks_data['metadata']['total_clips'] = len(landmarks_data['clips'])
    
    # Save updated landmarks file
    with open(landmarks_file, 'w', encoding='utf-8') as f:
        json.dump(landmarks_data, f, indent=2)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ Fixed {fixed_count}/{len(FAILED_CLIPS)} clips")
    logger.info(f"📊 Total clips with landmarks: {len(landmarks_data['clips'])}")
    logger.info(f"💾 Updated: {landmarks_file}")
    logger.info("=" * 80)
    
    return fixed_count == len(FAILED_CLIPS)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
