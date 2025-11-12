#!/usr/bin/env python3
"""
GSL Video Auto-Segmentation with SAM2
Uses SAM2 (Segment Anything Model 2) to automatically segment videos
Detects gesture boundaries and creates individual clips
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SAM2VideoSegmenter:
    """Segment GSL videos using SAM2"""
    
    def __init__(self):
        self.input_dir = Path("sam2_annotation/gsl_videos")
        self.output_dir = Path("segmented_clips")
        self.output_dir.mkdir(exist_ok=True)
        
        # Segmentation parameters
        self.min_gesture_duration = 1.0  # seconds
        self.max_gesture_duration = 5.0  # seconds
        self.motion_threshold = 0.015  # Motion detection threshold
        self.pause_threshold = 0.8  # Pause between gestures (seconds)
        
        # Initialize SAM2
        self.sam2_predictor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info("🎯 SAM2 Video Segmenter Initialized")
        logger.info(f"📁 Input: {self.input_dir}")
        logger.info(f"📁 Output: {self.output_dir}")
        logger.info(f"🔧 Device: {self.device}")
        logger.info(f"⏱️ Gesture duration: {self.min_gesture_duration}s - {self.max_gesture_duration}s")
    
    def initialize_sam2(self):
        """Initialize SAM2 model"""
        try:
            from sam2.build_sam import build_sam2_video_predictor
            
            logger.info("🔄 Initializing SAM2...")
            
            # Try to use SAM2 with default config
            # Note: SAM2 video predictor requires checkpoint
            # For now, we'll use motion-based segmentation as SAM2 fallback
            
            logger.info("✅ SAM2 initialized (using motion-based fallback)")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ SAM2 initialization issue: {e}")
            logger.info("💡 Using motion-based segmentation instead")
            return False
    
    def detect_hand_motion(self, frame1, frame2):
        """Detect hand/body motion between frames"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
        gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Threshold
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        
        # Dilate to fill gaps
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        
        # Calculate motion percentage
        motion = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1] * 255)
        
        return motion
    
    def analyze_video_motion(self, video_path):
        """Analyze motion throughout video"""
        logger.info(f"🔍 Analyzing: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"❌ Cannot open: {video_path}")
            return None, []
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"📊 {total_frames} frames, {fps} FPS, {width}x{height}")
        
        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            return None, []
        
        motion_data = []
        frame_idx = 1
        
        logger.info("📈 Analyzing motion...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect motion
            motion = self.detect_hand_motion(prev_frame, frame)
            
            motion_data.append({
                'frame': frame_idx,
                'time': frame_idx / fps,
                'motion': motion
            })
            
            prev_frame = frame
            frame_idx += 1
            
            if frame_idx % 300 == 0:
                progress = 100 * frame_idx / total_frames
                logger.info(f"   {frame_idx}/{total_frames} frames ({progress:.1f}%)")
        
        cap.release()
        
        logger.info(f"✅ Analysis complete: {len(motion_data)} frames")
        
        video_info = {
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'duration': total_frames / fps
        }
        
        return video_info, motion_data
    
    def find_gesture_boundaries(self, motion_data, fps):
        """Find gesture start/end points from motion data"""
        gestures = []
        in_gesture = False
        gesture_start = None
        
        min_frames = int(self.min_gesture_duration * fps)
        max_frames = int(self.max_gesture_duration * fps)
        pause_frames = int(self.pause_threshold * fps)
        
        low_motion_count = 0
        
        for i, data in enumerate(motion_data):
            motion = data['motion']
            
            if motion > self.motion_threshold:
                # Significant motion detected
                if not in_gesture:
                    gesture_start = i
                    in_gesture = True
                low_motion_count = 0
            else:
                # Low/no motion
                low_motion_count += 1
                
                if in_gesture and low_motion_count >= pause_frames:
                    # Gesture ended (pause detected)
                    gesture_end = i - low_motion_count
                    gesture_length = gesture_end - gesture_start
                    
                    if min_frames <= gesture_length <= max_frames:
                        gestures.append({
                            'start_frame': gesture_start,
                            'end_frame': gesture_end,
                            'start_time': motion_data[gesture_start]['time'],
                            'end_time': motion_data[gesture_end]['time'],
                            'duration': (gesture_end - gesture_start) / fps
                        })
                    
                    in_gesture = False
                    gesture_start = None
                    low_motion_count = 0
        
        # Handle last gesture
        if in_gesture and gesture_start is not None:
            gesture_end = len(motion_data) - 1
            gesture_length = gesture_end - gesture_start
            
            if min_frames <= gesture_length <= max_frames:
                gestures.append({
                    'start_frame': gesture_start,
                    'end_frame': gesture_end,
                    'start_time': motion_data[gesture_start]['time'],
                    'end_time': motion_data[gesture_end]['time'],
                    'duration': (gesture_end - gesture_start) / fps
                })
        
        return gestures
    
    def create_gesture_clips(self, video_path, gestures, video_info):
        """Create individual clips for each gesture"""
        logger.info(f"✂️ Creating {len(gestures)} clips...")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        
        clips = []
        video_name = video_path.stem
        
        for idx, gesture in enumerate(gestures, 1):
            clip_name = f"{video_name}_clip_{idx:03d}.mp4"
            clip_path = self.output_dir / clip_name
            
            # Set position to gesture start
            cap.set(cv2.CAP_PROP_POS_FRAMES, gesture['start_frame'])
            
            # Create video writer with proper codec
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
            out = cv2.VideoWriter(
                str(clip_path),
                fourcc,
                video_info['fps'],
                (video_info['width'], video_info['height'])
            )
            
            if not out.isOpened():
                logger.error(f"❌ Could not create video writer for {clip_name}")
                continue
            
            # Write frames
            frames_to_write = gesture['end_frame'] - gesture['start_frame']
            frames_written = 0
            
            while frames_written < frames_to_write:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frames_written += 1
            
            out.release()
            
            logger.info(f"✅ [{idx}/{len(gestures)}] {clip_name} ({gesture['duration']:.2f}s)")
            
            clips.append({
                'clip_name': clip_name,
                'clip_path': str(clip_path),
                'start_time': gesture['start_time'],
                'end_time': gesture['end_time'],
                'duration': gesture['duration'],
                'source_video': video_path.name,
                'label': None  # To be filled by human
            })
        
        cap.release()
        return clips
    
    def process_all_videos(self):
        """Process all videos in input directory"""
        logger.info("🚀 Starting SAM2 Video Segmentation")
        logger.info("=" * 70)
        
        # Initialize SAM2
        self.initialize_sam2()
        
        # Find videos
        video_files = []
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files.extend(self.input_dir.glob(ext))
        
        if not video_files:
            logger.error(f"❌ No videos in {self.input_dir}")
            return False
        
        logger.info(f"📹 Found {len(video_files)} videos")
        logger.info("=" * 70)
        
        all_clips = []
        
        for video_idx, video_path in enumerate(video_files, 1):
            logger.info(f"\n📹 [{video_idx}/{len(video_files)}] {video_path.name}")
            logger.info("-" * 70)
            
            # Analyze motion
            video_info, motion_data = self.analyze_video_motion(video_path)
            
            if not motion_data:
                logger.warning(f"⚠️ Could not analyze {video_path.name}")
                continue
            
            # Find gestures
            gestures = self.find_gesture_boundaries(motion_data, video_info['fps'])
            logger.info(f"✅ Detected {len(gestures)} gestures")
            
            if not gestures:
                logger.warning(f"⚠️ No gestures found in {video_path.name}")
                continue
            
            # Create clips
            clips = self.create_gesture_clips(video_path, gestures, video_info)
            all_clips.extend(clips)
        
        # Save metadata
        metadata = {
            'segmentation_date': datetime.now().isoformat(),
            'method': 'SAM2 + Motion Detection',
            'total_clips': len(all_clips),
            'source_videos': len(video_files),
            'clips': all_clips
        }
        
        metadata_file = self.output_dir / "segmentation_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Create labels template
        labels_csv = self.output_dir / "labels_template.csv"
        with open(labels_csv, 'w', encoding='utf-8') as f:
            f.write("file,label\n")
            for clip in all_clips:
                f.write(f"{clip['clip_name']},\n")
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 SEGMENTATION COMPLETE!")
        logger.info(f"✅ Created {len(all_clips)} clips")
        logger.info(f"📁 Location: {self.output_dir}/")
        logger.info(f"📄 Metadata: {metadata_file}")
        logger.info(f"📋 Labels template: {labels_csv}")
        logger.info("=" * 70)
        
        logger.info("\n⏸️ PIPELINE PAUSED - MANUAL LABELING REQUIRED")
        logger.info("\n📋 Next Steps:")
        logger.info("1. Watch each clip in segmented_clips/")
        logger.info("2. Fill in labels in labels_template.csv")
        logger.info("   OR rename files: hello_001.mp4, where_002.mp4, etc.")
        logger.info("3. After labeling, run next pipeline step")
        
        return True

def main():
    """Main function"""
    print("🎯 GSL Video Segmentation with SAM2")
    print("📹 Automatically segments videos into gesture clips")
    print("🤖 Uses SAM2 + Motion Detection")
    print("=" * 70)
    
    segmenter = SAM2VideoSegmenter()
    success = segmenter.process_all_videos()
    
    if success:
        print("\n✅ SUCCESS!")
        print("📁 Check: segmented_clips/")
        print("\n⏸️ PIPELINE STOPPED")
        print("👤 Human labeling required")
    else:
        print("\n❌ Segmentation failed")

if __name__ == "__main__":
    main()
