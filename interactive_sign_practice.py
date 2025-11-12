"""
Interactive GSL Sign Practice

Practice signing Ghanaian Sign Language words with real-time feedback.
The system will guide you through 5 words and evaluate your performance.
"""

import cv2
import mediapipe as mp
import time
import numpy as np
from datetime import datetime
import os

# Initialize MediaPipe Holistic for full-body tracking
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class GSLPracticeSession:
    """Interactive GSL practice session with full-body gesture feedback."""
    
    # 5 beginner-friendly GSL signs to practice
    PRACTICE_SIGNS = [
        {
            "word": "GOOD",
            "description": "Thumbs up gesture - extend thumb upward",
            "key_points": "Fist with thumb extended up, confident gesture",
            "difficulty": "Easy"
        },
        {
            "word": "LOVE",
            "description": "Cross both arms over your chest",
            "key_points": "Both hands on opposite shoulders, hugging motion",
            "difficulty": "Easy"
        },
        {
            "word": "HELP",
            "description": "Place one flat hand under the other fist, lift up",
            "key_points": "Bottom hand supports top fist, upward motion",
            "difficulty": "Medium"
        },
        {
            "word": "FRIEND",
            "description": "Hook index fingers together and shake",
            "key_points": "Index fingers linked, gentle shaking motion",
            "difficulty": "Medium"
        },
        {
            "word": "FAMILY",
            "description": "Make 'F' shape with both hands, circle around",
            "key_points": "Thumb and index touching, other fingers extended",
            "difficulty": "Medium"
        }
    ]
    
    def __init__(self):
        """Initialize practice session with full-body tracking."""
        # Use MediaPipe Holistic for full-body gesture recognition
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.current_sign_index = 0
        self.recording = False
        self.recording_start_time = None
        self.recorded_frames = []
        self.feedback_history = []
        
        # Create results directory
        self.results_dir = "practice_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
    
    def get_current_sign(self):
        """Get the current sign to practice."""
        if self.current_sign_index < len(self.PRACTICE_SIGNS):
            return self.PRACTICE_SIGNS[self.current_sign_index]
        return None
    
    def analyze_full_body_gesture(self, results):
        """
        Analyze full-body gesture including pose, hands, and face.
        
        This provides much better accuracy for GSL signs.
        """
        if not results.pose_landmarks and not results.left_hand_landmarks and not results.right_hand_landmarks:
            return {
                "detected": False,
                "confidence": 0.0,
                "feedback": "No person detected"
            }
        
        # Extract features from all body parts
        pose_features = self.extract_pose_features(results.pose_landmarks)
        hand_features = self.extract_hand_features(results.left_hand_landmarks, results.right_hand_landmarks)
        face_features = self.extract_face_features(results.face_landmarks)
        
        # Generate feedback based on current sign
        current_sign = self.get_current_sign()
        if not current_sign:
            return {"detected": True, "confidence": 0.0, "feedback": "Practice complete!"}
        
        feedback = self.generate_full_body_feedback(
            current_sign["word"],
            pose_features,
            hand_features,
            face_features,
            results
        )
        
        return feedback
    
    def extract_pose_features(self, pose_landmarks):
        """Extract pose features for body positioning"""
        if not pose_landmarks:
            return {
                "detected": False,
                "shoulder_width": 0,
                "arm_position": "unknown",
                "body_center_y": 0.5
            }
        
        landmarks = pose_landmarks.landmark
        
        # Key pose points
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        nose = landmarks[0]
        
        # Calculate features
        shoulder_width = abs(right_shoulder.x - left_shoulder.x)
        
        # Arm positions relative to body
        left_arm_raised = left_wrist.y < left_shoulder.y
        right_arm_raised = right_wrist.y < right_shoulder.y
        
        # Arms crossed (for LOVE sign)
        arms_crossed = (left_wrist.x > nose.x and right_wrist.x < nose.x) or \
                      (left_wrist.x < nose.x and right_wrist.x > nose.x)
        
        return {
            "detected": True,
            "shoulder_width": shoulder_width,
            "left_arm_raised": left_arm_raised,
            "right_arm_raised": right_arm_raised,
            "arms_crossed": arms_crossed,
            "body_center_y": nose.y,
            "left_wrist_y": left_wrist.y,
            "right_wrist_y": right_wrist.y
        }
    
    def extract_hand_features(self, left_hand, right_hand):
        """Extract hand features"""
        features = {
            "left_detected": left_hand is not None,
            "right_detected": right_hand is not None,
            "left_spread": 0,
            "right_spread": 0,
            "left_position": 0.5,
            "right_position": 0.5
        }
        
        if left_hand:
            features["left_spread"] = self.calculate_finger_spread(left_hand.landmark)
            features["left_position"] = left_hand.landmark[9].y
        
        if right_hand:
            features["right_spread"] = self.calculate_finger_spread(right_hand.landmark)
            features["right_position"] = right_hand.landmark[9].y
        
        return features
    
    def extract_face_features(self, face_landmarks):
        """Extract facial expression features"""
        if not face_landmarks:
            return {"detected": False, "expression": "neutral"}
        
        # Simple facial feature extraction
        # In production, you'd analyze smile, eyebrows, etc.
        return {
            "detected": True,
            "expression": "engaged"
        }
    
    def calculate_finger_spread(self, landmarks):
        """Calculate how spread out the fingers are."""
        thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
        index_tip = np.array([landmarks[8].x, landmarks[8].y])
        middle_tip = np.array([landmarks[12].x, landmarks[12].y])
        ring_tip = np.array([landmarks[16].x, landmarks[16].y])
        pinky_tip = np.array([landmarks[20].x, landmarks[20].y])
        
        # Calculate average distance between fingertips
        distances = [
            np.linalg.norm(index_tip - thumb_tip),
            np.linalg.norm(middle_tip - index_tip),
            np.linalg.norm(ring_tip - middle_tip),
            np.linalg.norm(pinky_tip - ring_tip)
        ]
        
        return np.mean(distances)
    
    def generate_full_body_feedback(self, sign_word, pose_features, hand_features, face_features, results):
        """Generate feedback based on full-body gesture analysis."""
        confidence = 0.0
        feedback_text = []
        
        if sign_word == "GOOD":
            # Check for thumbs up with arm raised
            if hand_features["right_detected"] or hand_features["left_detected"]:
                confidence += 0.3
                feedback_text.append("✓ Hand detected")
            
            # Check if arm is raised
            if pose_features["right_arm_raised"] or pose_features["left_arm_raised"]:
                confidence += 0.4
                feedback_text.append("✓ Arm raised - great!")
            else:
                feedback_text.append("✗ Raise your arm higher")
            
            # Check fist formation
            avg_spread = (hand_features["left_spread"] + hand_features["right_spread"]) / 2
            if avg_spread < 0.07:
                confidence += 0.3
                feedback_text.append("✓ Good thumbs up shape")
            
            
        elif sign_word == "LOVE":
            # Check for arms crossed on chest
            if pose_features["arms_crossed"]:
                confidence += 0.5
                feedback_text.append("✓ Arms crossed - perfect!")
            else:
                feedback_text.append("✗ Cross arms over chest")
            
            # Check both hands visible
            if hand_features["left_detected"] and hand_features["right_detected"]:
                confidence += 0.3
                feedback_text.append("✓ Both hands visible")
            
            # Check hand position on chest
            if 0.4 < hand_features["left_position"] < 0.7 or 0.4 < hand_features["right_position"] < 0.7:
                confidence += 0.2
                feedback_text.append("✓ Hands on chest area")
            
        elif sign_word == "HELP":
            # Check for both hands present
            if hand_features["left_detected"] and hand_features["right_detected"]:
                confidence += 0.4
                feedback_text.append("✓ Both hands detected")
            else:
                feedback_text.append("✗ Show both hands")
            
            # Check hand positioning (one supporting other)
            if 0.4 < hand_features["right_position"] < 0.7:
                confidence += 0.3
                feedback_text.append("✓ Good hand height")
            
            confidence += 0.3
            
        elif sign_word == "FRIEND":
            # Check for both hands at similar height
            if hand_features["left_detected"] and hand_features["right_detected"]:
                confidence += 0.4
                feedback_text.append("✓ Both hands visible")
                
                # Check if hands are close together (linking)
                hand_distance = abs(hand_features["left_position"] - hand_features["right_position"])
                if hand_distance < 0.2:
                    confidence += 0.4
                    feedback_text.append("✓ Hands linked together!")
                else:
                    feedback_text.append("✗ Bring hands closer")
            else:
                feedback_text.append("✗ Show both hands")
            
            confidence += 0.2
            
        elif sign_word == "FAMILY":
            # Check for F-shape with both hands
            if hand_features["left_detected"] and hand_features["right_detected"]:
                confidence += 0.4
                feedback_text.append("✓ Both hands forming F")
            else:
                feedback_text.append("✗ Use both hands")
            
            # Check finger spread for F-shape
            avg_spread = (hand_features["left_spread"] + hand_features["right_spread"]) / 2
            if 0.06 < avg_spread < 0.10:
                confidence += 0.3
                feedback_text.append("✓ Good F-shape")
            
            # Check hand position
            if 0.3 < hand_features["right_position"] < 0.6:
                confidence += 0.3
                feedback_text.append("✓ Good height")
        
        # Add body posture feedback
        if pose_features["detected"]:
            feedback_text.append("✓ Full body tracked")
        
        # Add facial expression feedback
        if face_features["detected"]:
            confidence += 0.1  # Bonus for facial engagement
        
        return {
            "detected": True,
            "confidence": min(confidence, 1.0),
            "feedback": " | ".join(feedback_text),
            "score": int(confidence * 100)
        }
    
    def draw_instruction_panel(self, frame):
        """Draw instruction panel on frame."""
        h, w, _ = frame.shape
        
        current_sign = self.get_current_sign()
        if not current_sign:
            return frame
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw sign information
        y_offset = 40
        
        # Sign number and word
        sign_text = f"Sign {self.current_sign_index + 1}/5: {current_sign['word']}"
        cv2.putText(frame, sign_text, (20, y_offset), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 3)
        y_offset += 40
        
        # Description
        cv2.putText(frame, current_sign['description'], (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 35
        
        # Key points
        cv2.putText(frame, f"Key: {current_sign['key_points']}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_offset += 35
        
        # Instructions
        if not self.recording:
            cv2.putText(frame, "Press SPACE to start recording (3 seconds)", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            elapsed = time.time() - self.recording_start_time
            remaining = max(0, 3.0 - elapsed)
            cv2.putText(frame, f"RECORDING... {remaining:.1f}s", (20, y_offset),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
        
        return frame
    
    def draw_feedback_panel(self, frame, feedback):
        """Draw feedback panel on frame."""
        h, w, _ = frame.shape
        
        # Draw feedback background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 150), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        if feedback and feedback.get("detected"):
            # Score bar
            score = feedback.get("score", 0)
            bar_width = int((w - 40) * (score / 100))
            
            # Color based on score
            if score >= 70:
                color = (0, 255, 0)  # Green
            elif score >= 40:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            cv2.rectangle(frame, (20, h - 130), (20 + bar_width, h - 110), color, -1)
            cv2.rectangle(frame, (20, h - 130), (w - 20, h - 110), (255, 255, 255), 2)
            
            # Score text
            cv2.putText(frame, f"Score: {score}%", (20, h - 140),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            
            # Feedback text
            feedback_text = feedback.get("feedback", "")
            y_offset = h - 80
            
            # Split long feedback into multiple lines
            max_chars = 80
            words = feedback_text.split(" | ")
            for word in words:
                cv2.putText(frame, word, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
        
        return frame
    
    def draw_controls(self, frame):
        """Draw control instructions."""
        h, w, _ = frame.shape
        
        controls = [
            "SPACE: Record sign",
            "N: Next sign",
            "R: Repeat current",
            "Q: Quit"
        ]
        
        x_offset = w - 250
        y_offset = h - 120
        
        for control in controls:
            cv2.putText(frame, control, (x_offset, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25
        
        return frame
    
    def save_practice_result(self, sign_word, score, feedback):
        """Save practice result."""
        result = {
            "sign": sign_word,
            "score": score,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }
        self.feedback_history.append(result)
    
    def show_final_results(self):
        """Display final practice results."""
        print("\n" + "=" * 60)
        print("PRACTICE SESSION COMPLETE!")
        print("=" * 60)
        
        if not self.feedback_history:
            print("No signs practiced.")
            return
        
        total_score = 0
        print("\nYour Performance:\n")
        
        for i, result in enumerate(self.feedback_history, 1):
            score = result["score"]
            total_score += score
            
            # Performance indicator
            if score >= 70:
                indicator = "🌟 Excellent!"
            elif score >= 50:
                indicator = "👍 Good!"
            elif score >= 30:
                indicator = "📚 Keep practicing"
            else:
                indicator = "💪 Needs work"
            
            print(f"{i}. {result['sign']}: {score}% {indicator}")
            print(f"   {result['feedback']}\n")
        
        avg_score = total_score / len(self.feedback_history)
        print("=" * 60)
        print(f"AVERAGE SCORE: {avg_score:.1f}%")
        
        if avg_score >= 70:
            print("🎉 Outstanding! You're doing great!")
        elif avg_score >= 50:
            print("👏 Good job! Keep practicing to improve!")
        else:
            print("💪 Keep practicing! You'll get better!")
        
        print("=" * 60)
    
    def run(self, camera_index=0):
        """Run interactive practice session."""
        print("\n" + "=" * 60)
        print("GSL INTERACTIVE PRACTICE SESSION")
        print("FULL-BODY GESTURE RECOGNITION (Pose + Hands + Face)")
        print("=" * 60)
        print("\nYou will practice 5 basic GSL signs:")
        for i, sign in enumerate(self.PRACTICE_SIGNS, 1):
            print(f"{i}. {sign['word']} - {sign['description']}")
        
        print("\nControls:")
        print("  SPACE - Record your sign (3 seconds)")
        print("  N - Next sign")
        print("  R - Repeat current sign")
        print("  Q - Quit")
        print("\nStarting camera...")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("✗ Error: Could not open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("✓ Camera ready! Show your signs!\n")
        
        current_feedback = None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(rgb_frame)
                
                # Draw pose landmarks
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                
                # Draw hand landmarks
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                # Draw face landmarks (subtle)
                if results.face_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        results.face_landmarks,
                        mp_holistic.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                
                # Analyze full-body gesture if recording
                if self.recording:
                    current_feedback = self.analyze_full_body_gesture(results)
                
                # Check recording timeout
                if self.recording:
                    elapsed = time.time() - self.recording_start_time
                    if elapsed >= 3.0:
                        self.recording = False
                        if current_feedback:
                            current_sign = self.get_current_sign()
                            self.save_practice_result(
                                current_sign["word"],
                                current_feedback.get("score", 0),
                                current_feedback.get("feedback", "")
                            )
                            print(f"✓ {current_sign['word']}: {current_feedback.get('score', 0)}%")
                
                # Draw UI
                frame = self.draw_instruction_panel(frame)
                frame = self.draw_feedback_panel(frame, current_feedback)
                frame = self.draw_controls(frame)
                
                cv2.imshow('GSL Practice Session', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' ') and not self.recording:
                    self.recording = True
                    self.recording_start_time = time.time()
                    current_feedback = None
                elif key == ord('n'):
                    if self.current_sign_index < len(self.PRACTICE_SIGNS) - 1:
                        self.current_sign_index += 1
                        current_feedback = None
                        print(f"\nNext sign: {self.get_current_sign()['word']}")
                    else:
                        print("\nAll signs completed! Press Q to see results.")
                elif key == ord('r'):
                    current_feedback = None
                    print(f"\nRepeating: {self.get_current_sign()['word']}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.holistic.close()
            self.show_final_results()


def main():
    """Main function."""
    session = GSLPracticeSession()
    session.run(camera_index=0)


if __name__ == "__main__":
    main()
