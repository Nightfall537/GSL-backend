"""
Real-time Hand Detection with Webcam

This script uses MediaPipe to detect hands in real-time from your laptop webcam.
Press 'q' to quit, 's' to save a screenshot.
"""

import cv2
import mediapipe as mp
import time
from datetime import datetime
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class HandDetector:
    """Real-time hand detection using webcam."""
    
    def __init__(self, 
                 max_num_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        Initialize hand detector.
        
        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.fps_history = []
        self.screenshot_dir = "screenshots"
        
        # Create screenshots directory
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir)
    
    def process_frame(self, frame):
        """
        Process a single frame for hand detection.
        
        Args:
            frame: BGR image from webcam
            
        Returns:
            Processed frame with hand landmarks drawn
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Get hand label (Left/Right)
                if results.multi_handedness:
                    handedness = results.multi_handedness[hand_idx]
                    hand_label = handedness.classification[0].label
                    hand_score = handedness.classification[0].score
                    
                    # Draw hand label
                    h, w, _ = frame.shape
                    cx = int(hand_landmarks.landmark[0].x * w)
                    cy = int(hand_landmarks.landmark[0].y * h)
                    
                    cv2.putText(
                        frame,
                        f"{hand_label} ({hand_score:.2f})",
                        (cx - 50, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
        
        return frame, results
    
    def calculate_fps(self, start_time):
        """Calculate and return current FPS."""
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = 1.0 / elapsed
            self.fps_history.append(fps)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            return sum(self.fps_history) / len(self.fps_history)
        return 0
    
    def draw_info(self, frame, fps, num_hands):
        """Draw information overlay on frame."""
        h, w, _ = frame.shape
        
        # Draw semi-transparent background for info
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw text info
        info_text = [
            f"FPS: {fps:.1f}",
            f"Hands Detected: {num_hands}",
            "Press 'q' to quit",
            "Press 's' to save screenshot"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(
                frame,
                text,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            y_offset += 25
        
        return frame
    
    def save_screenshot(self, frame):
        """Save current frame as screenshot."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.screenshot_dir, f"hand_detection_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved: {filename}")
        return filename
    
    def run(self, camera_index=0):
        """
        Run real-time hand detection.
        
        Args:
            camera_index: Camera device index (0 for default webcam)
        """
        print("=" * 60)
        print("REAL-TIME HAND DETECTION")
        print("=" * 60)
        print(f"\nInitializing camera {camera_index}...")
        
        # Open webcam
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("✗ Error: Could not open webcam")
            print("\nTroubleshooting:")
            print("1. Check if your webcam is connected")
            print("2. Try a different camera_index (0, 1, 2, etc.)")
            print("3. Check if another application is using the webcam")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✓ Camera opened successfully")
        print(f"  Resolution: {actual_width}x{actual_height}")
        print(f"\nStarting hand detection...")
        print("  - Show your hands to the camera")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save screenshot")
        print()
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                frame_start = time.time()
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("✗ Error: Failed to read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                processed_frame, results = self.process_frame(frame)
                
                # Calculate FPS
                fps = self.calculate_fps(frame_start)
                
                # Count hands
                num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
                
                # Draw info overlay
                processed_frame = self.draw_info(processed_frame, fps, num_hands)
                
                # Display frame
                cv2.imshow('Hand Detection - GSL Backend', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    self.save_screenshot(processed_frame)
                
                frame_count += 1
                
                # Print stats every 100 frames
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    avg_fps = frame_count / elapsed
                    print(f"Processed {frame_count} frames | Avg FPS: {avg_fps:.1f}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            # Cleanup
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            
            print("\n" + "=" * 60)
            print("SESSION SUMMARY")
            print("=" * 60)
            print(f"Total frames processed: {frame_count}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Average FPS: {avg_fps:.1f}")
            print(f"Screenshots saved in: {self.screenshot_dir}/")
            print("=" * 60)
            
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()


def test_camera_availability():
    """Test which camera indices are available."""
    print("Testing camera availability...")
    available_cameras = []
    
    for i in range(5):  # Test first 5 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"  Camera {i}: Available")
            cap.release()
    
    if not available_cameras:
        print("  No cameras found!")
    
    return available_cameras


def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("GSL BACKEND - WEBCAM HAND DETECTION TEST")
    print("=" * 60)
    
    # Test camera availability
    available_cameras = test_camera_availability()
    
    if not available_cameras:
        print("\n✗ No cameras detected!")
        print("\nPlease check:")
        print("1. Your webcam is connected")
        print("2. Camera permissions are granted")
        print("3. No other application is using the camera")
        return
    
    print(f"\nFound {len(available_cameras)} camera(s): {available_cameras}")
    
    # Use first available camera
    camera_index = available_cameras[0]
    print(f"\nUsing camera {camera_index}")
    
    # Create detector
    detector = HandDetector(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Run detection
    detector.run(camera_index=camera_index)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
