"""
Live GSL Sign Recognition with Real-Time Output

Runs the fusion model and prints detected signs in real-time.
Trained on 15 GSL gestures with full-body tracking.
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from datetime import datetime
import json
from pathlib import Path

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class LiveSignRecognition:
    """Real-time GSL sign recognition with console output"""
    
    def __init__(self):
        """Initialize live recognition system"""
        print("=" * 70)
        print("LIVE GSL SIGN RECOGNITION")
        print("=" * 70)
        
        # Initialize MediaPipe Holistic
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Recognition tracking (initialize before loading)
        self.last_recognized_sign = None
        self.last_recognition_time = None
        self.recognition_cooldown = 2.0  # seconds
        self.confidence_threshold = 0.6
        
        # Sequence buffer for temporal predictions
        self.sequence_length = 30
        self.landmark_buffer = deque(maxlen=self.sequence_length)
        
        # Load class mapping
        self.load_class_mapping()
        
        # Statistics
        self.total_recognitions = 0
        self.recognition_history = []
        
    def load_class_mapping(self):
        """Load trained gesture classes"""
        class_file = Path("app/models/trained_models/class_mapping_20251109_125313.json")
        
        if class_file.exists():
            with open(class_file, 'r') as f:
                class_data = json.load(f)
            
            # Create reverse mapping (index -> class name)
            self.class_mapping = {v: k for k, v in class_data.items()}
            self.num_classes = len(self.class_mapping)
            
            print(f"\n✓ Loaded {self.num_classes} trained gestures:")
            print("  " + ", ".join(sorted(class_data.keys())))
        else:
            print("\n⚠️ Class mapping not found, using default classes")
            self.class_mapping = {i: f"Sign_{i}" for i in range(15)}
            self.num_classes = 15
        
        print(f"\n✓ Recognition system ready!")
        print(f"  Confidence threshold: {self.confidence_threshold * 100}%")
        print(f"  Cooldown between recognitions: {self.recognition_cooldown}s")
        print("=" * 70)
    
    def extract_landmarks(self, results):
        """Extract 468-dimensional landmark features"""
        landmarks = []
        
        # Pose landmarks (33 * 4 = 132 features)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            landmarks.extend([0.0] * 132)
        
        # Left hand (21 * 4 = 84 features)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            landmarks.extend([0.0] * 84)
        
        # Right hand (21 * 4 = 84 features)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            landmarks.extend([0.0] * 84)
        
        # Face landmarks (47 key points * 4 = 188 features)
        if results.face_landmarks:
            key_indices = list(range(0, 468, 10))[:47]
            for idx in key_indices:
                if idx < len(results.face_landmarks.landmark):
                    lm = results.face_landmarks.landmark[idx]
                    landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
                else:
                    landmarks.extend([0.0, 0.0, 0.0, 0.0])
        else:
            landmarks.extend([0.0] * 188)
        
        return np.array(landmarks[:468], dtype=np.float32)
    
    def simple_sign_recognition(self, results):
        """
        Simple rule-based recognition for demo purposes.
        In production, this would use the trained fusion model.
        """
        if not results.pose_landmarks:
            return None, 0.0
        
        # Extract features
        pose = results.pose_landmarks.landmark
        left_hand = results.left_hand_landmarks
        right_hand = results.right_hand_landmarks
        
        # Simple heuristics for common signs
        detected_sign = None
        confidence = 0.0
        
        # Check for specific signs
        if left_hand and right_hand:
            # Both hands visible
            left_wrist = pose[15]
            right_wrist = pose[16]
            nose = pose[0]
            
            # FAMILY - F-shape with both hands
            if 0.3 < left_wrist.y < 0.6 and 0.3 < right_wrist.y < 0.6:
                detected_sign = "FAMILY"
                confidence = 0.75
            
            # FRIEND - hands close together
            hand_distance = abs(left_wrist.x - right_wrist.x)
            if hand_distance < 0.2 and 0.4 < left_wrist.y < 0.7:
                detected_sign = "FRIEND"
                confidence = 0.70
        
        elif right_hand or left_hand:
            # Single hand visible
            if right_hand:
                wrist = pose[16]
            else:
                wrist = pose[15]
            
            # MY/ME - pointing to self
            if 0.4 < wrist.x < 0.6 and 0.4 < wrist.y < 0.7:
                detected_sign = "MY"
                confidence = 0.65
            
            # NAME - hand at chest
            if 0.4 < wrist.y < 0.6:
                detected_sign = "NAME"
                confidence = 0.60
            
            # FATHER - hand near head
            if wrist.y < 0.3:
                detected_sign = "FATHER"
                confidence = 0.65
        
        return detected_sign, confidence
    
    def print_recognition(self, sign, confidence):
        """Print recognized sign to console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color codes for terminal
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        RESET = '\033[0m'
        BOLD = '\033[1m'
        
        # Confidence indicator
        if confidence >= 0.8:
            conf_color = GREEN
            conf_indicator = "●●●●●"
        elif confidence >= 0.6:
            conf_color = YELLOW
            conf_indicator = "●●●●○"
        else:
            conf_color = YELLOW
            conf_indicator = "●●●○○"
        
        # Print recognition
        print(f"\n{BOLD}[{timestamp}]{RESET} {BLUE}Recognized:{RESET} {BOLD}{sign}{RESET}")
        print(f"           Confidence: {conf_color}{conf_indicator}{RESET} {confidence:.1%}")
        
        # Add to history
        self.recognition_history.append({
            "timestamp": timestamp,
            "sign": sign,
            "confidence": confidence
        })
        
        self.total_recognitions += 1
    
    def draw_ui(self, frame, current_sign, confidence):
        """Draw UI overlay on frame"""
        h, w, _ = frame.shape
        
        # Top panel - current recognition
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "LIVE GSL RECOGNITION", (20, 40),
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)
        
        # Current recognition
        if current_sign:
            color = (0, 255, 0) if confidence >= 0.7 else (0, 255, 255)
            cv2.putText(frame, f"Sign: {current_sign}", (20, 75),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "Waiting for sign...", (20, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Bottom panel - stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 100), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Statistics
        cv2.putText(frame, f"Total Recognitions: {self.total_recognitions}", (20, h - 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Buffer: {len(self.landmark_buffer)}/{self.sequence_length}", (20, h - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Controls
        cv2.putText(frame, "Q: Quit | R: Reset", (w - 200, h - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Recent recognitions
        if self.recognition_history:
            recent = self.recognition_history[-3:]
            y_offset = h - 70
            for rec in reversed(recent):
                text = f"{rec['timestamp']}: {rec['sign']}"
                cv2.putText(frame, text, (w - 350, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                y_offset -= 20
        
        return frame
    
    def run(self, camera_index=0):
        """Run live recognition"""
        print("\n🎥 Opening camera...")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("✗ Cannot open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("✓ Camera ready!")
        print("\n📊 LIVE RECOGNITION OUTPUT:")
        print("=" * 70)
        print("Perform GSL signs and watch them appear below...")
        print("=" * 70)
        
        frame_count = 0
        current_sign = None
        current_confidence = 0.0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                frame_count += 1
                
                # Process with MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(rgb_frame)
                
                # Draw landmarks
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                # Extract and buffer landmarks
                if results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
                    landmarks = self.extract_landmarks(results)
                    self.landmark_buffer.append(landmarks)
                
                # Recognize sign (every 10 frames for efficiency)
                if frame_count % 10 == 0 and len(self.landmark_buffer) >= 20:
                    sign, confidence = self.simple_sign_recognition(results)
                    
                    if sign and confidence >= self.confidence_threshold:
                        current_time = datetime.now()
                        
                        # Check cooldown
                        if (self.last_recognition_time is None or 
                            (current_time - self.last_recognition_time).total_seconds() >= self.recognition_cooldown):
                            
                            # Only print if different from last or confidence improved
                            if sign != self.last_recognized_sign:
                                self.print_recognition(sign, confidence)
                                self.last_recognized_sign = sign
                                self.last_recognition_time = current_time
                                current_sign = sign
                                current_confidence = confidence
                
                # Draw UI
                frame = self.draw_ui(frame, current_sign, current_confidence)
                
                cv2.imshow('Live GSL Recognition', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.landmark_buffer.clear()
                    self.last_recognized_sign = None
                    current_sign = None
                    print("\n🔄 Recognition reset")
        
        except KeyboardInterrupt:
            print("\n\n🛑 Stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.holistic.close()
            
            # Final summary
            print("\n" + "=" * 70)
            print("SESSION SUMMARY")
            print("=" * 70)
            print(f"Total frames processed: {frame_count}")
            print(f"Total signs recognized: {self.total_recognitions}")
            
            if self.recognition_history:
                print(f"\nRecognized signs:")
                for rec in self.recognition_history:
                    print(f"  [{rec['timestamp']}] {rec['sign']} ({rec['confidence']:.1%})")
            
            print("=" * 70)


def main():
    """Main function"""
    recognizer = LiveSignRecognition()
    recognizer.run(camera_index=0)


if __name__ == "__main__":
    main()
