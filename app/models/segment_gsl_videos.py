#!/usr/bin/env python3
"""
GSL Video Auto-Segmentation
Automatically segments long GSL videos into individual gesture clips
Uses motion detection and scene changes to find gesture boundaries
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GSLVideoSegmenter:
    """Automatically segment GSL videos into individual gesture clips"""
    
    def __init__(self):
        self.input_dir = Path("sam2_annotation/gsl_videos")
        self.output_dir = Path("segmented_clips")
        self.output_dir.mkdir(exist_ok=True)
        
        # Segmentation parameters
        self.min_gesture_duration = 1.0  # seconds
        self.max_gesture_duration = 5.0  # seconds
        self.motion_threshold = 0.02  # Motion detection threshold
        self.pause_threshold = 0.5  # Pause between gestures (seconds)
        
        logger.info("🎯 GSL Video Segmenter Initialized")
        logger.info(f"📁 Input: {self.input_dir}")
        logger.info(f"📁 Output: {self.output_dir}")
        logger.info(f"⏱️ Min gesture duration: {self.min_gesture_duration}s")
        logger.info(f"⏱️ Max gesture duration: {self.max_gesture_duration}s")
    
    def detect_motion(self, frame1, frame2):
        """Detect motion between two frames"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Threshold
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Calculate motion percentage
        motion = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1] * 255)
        
        return motion
    
    def find_gesture_boundaries(self, video_path):
        """Find gesture start/end boundaries in video"""
        logger.info(f"🔍 Analyzing video: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"❌ Cannot open video: {video_path}")
            return []
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"📊 Video: {total_frames} frames, {fps} FPS")
        
        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            return []
        
        motion_history = []
        frame_idx = 1
        
        # Analyze motion throughout video
        logger.info("📈 Analyzing motion...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            motion = self.detect_motion(prev_frame, frame)
            motion_history.append({
                'frame': frame_idx,
                'time': frame_idx / fps,
                'motion': motion
            })
            
            prev_frame = frame
            frame_idx += 1
            
            if frame_idx % 300 == 0:
                logger.info(f"   Processed {frame_idx}/{total_frames} frames ({100*frame_idx/total_frames:.1f}%)")
        
        cap.release()
        
        logger.info(f"✅ Motion analysis complete: {len(motion_history)} frames")
        
        # Find gesture boundaries based on motion
        gestures = self.extract_gesture_segments(motion_history, fps)
        
        logger.info(f"✅ Found {len(gestures)} potential gestures")
        
        return gestures
    
    def extract_gesture_segments(self, motion_history, fps):
        """Extract gesture segments from motion history"""
        gestures = []
        in_gesture = False
        gesture_start = None
        
        min_frames = int(self.min_gesture_duration * fps)
        max_frames = int(self.max_gesture_duration * fps)
        pause_frames = int(self.pause_threshold * fps)
        
        low_motion_count = 0
        
        for i, data in enumerate(motion_history):
            motion = data['motion']
            
            if motion > self.motion_threshold:
                # Motion detected
                if not in_gesture:
                    # Start of new gesture
                    gesture_start = i
                    in_gesture = True
                low_motion_count = 0
            else:
                # Low/no motion
                low_motion_count += 1
                
                if in_gesture and low_motion_count >= pause_frames:
                    # End of gesture (pause detected)
                    gesture_end = i - low_motion_count
                    gesture_length = gesture_end - gesture_start
                    
                    # Check if gesture duration is valid
                    if min_frames <= gesture_length <= max_frames:
                        gestures.append({
                            'start_frame': gesture_start,
                            'end_frame': gesture_end,
                            'start_time': motion_history[gesture_start]['time'],
                            'end_time': motion_history[gesture_end]['time'],
                            'duration': (gesture_end - gesture_start) / fps
                        })
                    
                    in_gesture = False
                    gesture_start = None
                    low_motion_count = 0
        
        # Handle last gesture if video ends during motion
        if in_gesture and gesture_start is not None:
            gesture_end = len(motion_history) - 1
            gesture_length = gesture_end - gesture_start
            
            if min_frames <= gesture_length <= max_frames:
                gestures.append({
                    'start_frame': gesture_start,
                    'end_frame': gesture_end,
                    'start_time': motion_history[gesture_start]['time'],
                    'end_time': motion_history[gesture_end]['time'],
                    'duration': (gesture_end - gesture_start) / fps
                })
        
        return gestures
    
    def segment_video(self, video_path, gestures):
        """Segment video into individual gesture clips"""
        logger.info(f"✂️ Segmenting video into {len(gestures)} clips...")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        segmented_clips = []
        
        for idx, gesture in enumerate(gestures, 1):
            # Create output filename
            video_name = video_path.stem
            clip_name = f"{video_name}_gesture_{idx:03d}.mp4"
            clip_path = self.output_dir / clip_name
            
            # Set video position to gesture start
            cap.set(cv2.CAP_PROP_POS_FRAMES, gesture['start_frame'])
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
            
            # Write frames for this gesture
            frames_to_write = gesture['end_frame'] - gesture['start_frame']
            frames_written = 0
            
            while frames_written < frames_to_write:
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
                frames_written += 1
            
            out.release()
            
            logger.info(f"✅ [{idx}/{len(gestures)}] Created: {clip_name} ({gesture['duration']:.2f}s)")
            
            segmented_clips.append({
                'clip_name': clip_name,
                'clip_path': str(clip_path),
                'start_time': gesture['start_time'],
                'end_time': gesture['end_time'],
                'duration': gesture['duration'],
                'source_video': video_path.name
            })
        
        cap.release()
        
        return segmented_clips
    
    def process_all_videos(self):
        """Process all videos in input directory"""
        logger.info("🚀 Starting GSL Video Segmentation")
        logger.info("=" * 70)
        
        # Find all videos
        video_files = list(self.input_dir.glob("*.mp4"))
        video_files.extend(self.input_dir.glob("*.avi"))
        video_files.extend(self.input_dir.glob("*.mov"))
        
        if not video_files:
            logger.error(f"❌ No videos found in {self.input_dir}")
            return False
        
        logger.info(f"📹 Found {len(video_files)} videos to process")
        logger.info("=" * 70)
        
        all_clips = []
        
        for video_idx, video_path in enumerate(video_files, 1):
            logger.info(f"\n📹 [{video_idx}/{len(video_files)}] Processing: {video_path.name}")
            logger.info("-" * 70)
            
            # Find gesture boundaries
            gestures = self.find_gesture_boundaries(video_path)
            
            if not gestures:
                logger.warning(f"⚠️ No gestures detected in {video_path.name}")
                continue
            
            # Segment video
            clips = self.segment_video(video_path, gestures)
            all_clips.extend(clips)
        
        # Save segmentation metadata
        metadata = {
            'segmentation_date': datetime.now().isoformat(),
            'total_clips': len(all_clips),
            'source_videos': len(video_files),
            'clips': all_clips
        }
        
        metadata_file = self.output_dir / "segmentation_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("\n" + "=" * 70)
        logger.info("🎉 Segmentation Complete!")
        logger.info(f"✅ Total clips created: {len(all_clips)}")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"📄 Metadata saved: {metadata_file}")
        logger.info("=" * 70)
        
        logger.info("\n⏸️ PIPELINE PAUSED")
        logger.info("📋 Next Step: MANUAL LABELING")
        logger.info("\nOptions for labeling:")
        logger.info("1. Rename clips: hello_001.mp4, where_002.mp4, etc.")
        logger.info("2. Create labels.csv with columns: file, label")
        logger.info("3. Create labels.json: {'file': 'clip.mp4', 'label': 'word'}")
        logger.info("\n💡 After labeling, run the next pipeline step")
        
        return True

def main():
    """Main function"""
    print("🎯 GSL Video Auto-Segmentation")
    print("📹 Segments long videos into individual gesture clips")
    print("🤖 Uses motion detection to find gesture boundaries")
    print("=" * 70)
    
    segmenter = GSLVideoSegmenter()
    success = segmenter.process_all_videos()
    
    if success:
        print("\n✅ SUCCESS! Videos segmented")
        print(f"📁 Check: segmented_clips/")
        print("\n⏸️ PIPELINE STOPPED")
        print("👤 Human labeling required before continuing")
    else:
        print("\n❌ Segmentation failed")

if __name__ == "__main__":
    main()
