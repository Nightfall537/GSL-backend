"""
GSL Phrase Practice with Full-Body Gesture Recognition

Practice complete phrases in Ghanaian Sign Language with real-time feedback.
Uses MediaPipe Holistic for pose + hands + face tracking.
"""

import cv2
import mediapipe as mp
import time
import numpy as np
from datetime import datetime
import os

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class GSLPhrasePracticeFullBody:
    """Practice GSL phrases with full-body gesture recognition"""
    
    # 5 practical GSL phrases (10 seconds each for comfortable practice)
    PRACTICE_PHRASES = [
        {
            "phrase": "HELLO MY NAME",
            "signs": ["HELLO", "MY", "NAME"],
            "description": "Wave hand → Point to chest → Touch fingertips together",
            "tips": "Smooth transitions, maintain eye contact, clear gestures",
            "difficulty": "Easy",
            "duration": 10
        },
        {
            "phrase": "THANK YOU FRIEND",
            "signs": ["THANK YOU", "FRIEND"],
            "description": "Hand from chin forward → Hook index fingers together",
            "tips": "Show gratitude with facial expression, gentle linking motion",
            "difficulty": "Easy",
            "duration": 10
        },
        {
            "phrase": "I LOVE MY FAMILY",
            "signs": ["ME", "LOVE", "MY", "FAMILY"],
            "description": "Point to self → Cross arms on chest → Point to chest → F-shape circle",
            "tips": "Show emotion, smooth flow between signs, use both hands",
            "difficulty": "Medium",
            "duration": 10
        },
        {
            "phrase": "PLEASE HELP ME",
            "signs": ["PLEASE", "HELP", "ME"],
            "description": "Circular motion on chest → One hand supports other → Point to self",
            "tips": "Polite facial expression, clear hand positioning, show need",
            "difficulty": "Medium",
            "duration": 10
        },
        {
            "phrase": "GOOD MORNING FATHER",
            "signs": ["GOOD", "MORNING", "FATHER"],
            "description": "Thumbs up → Sun rising gesture → Touch forehead then chin",
            "tips": "Respectful demeanor, clear morning gesture, proper sequence",
            "difficulty": "Medium",
            "duration": 10
        }
    ]
    
    def __init__(self):
        """Initialize full-body phrase practice"""
        self.holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.current_phrase_index = 0
        self.current_sign_index = 0
        self.recording = False
        self.recording_start_time = None
        self.feedback_history = []
        
        # Results directory
        self.results_dir = "phrase_practice_fullbody"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        # Sign detection tracking
        self.sign_detected_time = {}
        self.phrase_start_time = None
    
    def get_current_phrase(self):
        """Get current phrase"""
        if self.current_phrase_index < len(self.PRACTICE_PHRASES):
            return self.PRACTICE_PHRASES[self.current_phrase_index]
        return None
    
    def get_current_sign(self):
        """Get current sign in phrase"""
        phrase = self.get_current_phrase()
        if phrase and self.current_sign_index < len(phrase["signs"]):
            return phrase["signs"][self.current_sign_index]
        return None
    
    def extract_pose_features(self, pose_landmarks):
        """Extract pose features"""
        if not pose_landmarks:
            return {"detected": False}
        
        landmarks = pose_landmarks.landmark
        
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        nose = landmarks[0]
        
        return {
            "detected": True,
            "left_arm_raised": left_wrist.y < left_shoulder.y,
            "right_arm_raised": right_wrist.y < right_shoulder.y,
            "arms_crossed": (left_wrist.x > nose.x and right_wrist.x < nose.x) or 
                           (left_wrist.x < nose.x and right_wrist.x > nose.x),
            "left_wrist_y": left_wrist.y,
            "right_wrist_y": right_wrist.y,
            "left_wrist_x": left_wrist.x,
            "right_wrist_x": right_wrist.x,
            "nose_y": nose.y
        }
    
    def extract_hand_features(self, left_hand, right_hand):
        """Extract hand features"""
        features = {
            "left_detected": left_hand is not None,
            "right_detected": right_hand is not None,
            "left_spread": 0,
            "right_spread": 0,
            "left_position_y": 0.5,
            "right_position_y": 0.5,
            "left_position_x": 0.5,
            "right_position_x": 0.5
        }
        
        if left_hand:
            features["left_spread"] = self.calculate_finger_spread(left_hand.landmark)
            features["left_position_y"] = left_hand.landmark[9].y
            features["left_position_x"] = left_hand.landmark[9].x
        
        if right_hand:
            features["right_spread"] = self.calculate_finger_spread(right_hand.landmark)
            features["right_position_y"] = right_hand.landmark[9].y
            features["right_position_x"] = right_hand.landmark[9].x
        
        return features
    
    def calculate_finger_spread(self, landmarks):
        """Calculate finger spread"""
        thumb_tip = np.array([landmarks[4].x, landmarks[4].y])
        index_tip = np.array([landmarks[8].x, landmarks[8].y])
        middle_tip = np.array([landmarks[12].x, landmarks[12].y])
        ring_tip = np.array([landmarks[16].x, landmarks[16].y])
        pinky_tip = np.array([landmarks[20].x, landmarks[20].y])
        
        distances = [
            np.linalg.norm(index_tip - thumb_tip),
            np.linalg.norm(middle_tip - index_tip),
            np.linalg.norm(ring_tip - middle_tip),
            np.linalg.norm(pinky_tip - ring_tip)
        ]
        
        return np.mean(distances)
    
    def analyze_sign(self, sign_name, pose_features, hand_features):
        """Analyze specific sign"""
        confidence = 0.0
        feedback = []
        
        if sign_name == "HELLO":
            if hand_features["right_detected"] or hand_features["left_detected"]:
                confidence += 0.3
                feedback.append("✓ Hand visible")
            
            avg_spread = (hand_features["left_spread"] + hand_features["right_spread"]) / 2
            if avg_spread > 0.08:
                confidence += 0.4
                feedback.append("✓ Open palm - good wave!")
            else:
                feedback.append("✗ Spread fingers more")
            
            if pose_features["right_arm_raised"] or pose_features["left_arm_raised"]:
                confidence += 0.3
                feedback.append("✓ Hand raised")
        
        elif sign_name == "MY" or sign_name == "ME":
            if 0.4 < hand_features["right_position_x"] < 0.6:
                confidence += 0.5
                feedback.append("✓ Pointing to self")
            else:
                feedback.append("✗ Point to your chest")
            
            if 0.4 < hand_features["right_position_y"] < 0.7:
                confidence += 0.3
                feedback.append("✓ Good chest level")
            
            confidence += 0.2
        
        elif sign_name == "NAME":
            if hand_features["left_detected"] and hand_features["right_detected"]:
                confidence += 0.4
                feedback.append("✓ Both hands visible")
                
                # Check if fingertips are touching
                hand_distance = abs(hand_features["left_position_x"] - hand_features["right_position_x"])
                if hand_distance < 0.15:
                    confidence += 0.4
                    feedback.append("✓ Fingertips together!")
                else:
                    feedback.append("✗ Bring fingertips together")
            else:
                feedback.append("✗ Show both hands")
            
            confidence += 0.2
        
        elif sign_name == "THANK YOU":
            if hand_features["right_detected"]:
                confidence += 0.3
                feedback.append("✓ Hand detected")
            
            # Check hand near face/chin
            if hand_features["right_position_y"] < 0.4:
                confidence += 0.4
                feedback.append("✓ Hand near chin - perfect!")
            else:
                feedback.append("✗ Move hand to chin area")
            
            confidence += 0.3
        
        elif sign_name == "FRIEND":
            if hand_features["left_detected"] and hand_features["right_detected"]:
                confidence += 0.4
                feedback.append("✓ Both hands visible")
                
                # Check if hands are close (linking)
                hand_distance = abs(hand_features["left_position_x"] - hand_features["right_position_x"])
                if hand_distance < 0.2:
                    confidence += 0.5
                    feedback.append("✓ Fingers linked - excellent!")
                else:
                    feedback.append("✗ Hook fingers together")
            else:
                feedback.append("✗ Show both hands")
            
            confidence += 0.1
        
        elif sign_name == "LOVE":
            if pose_features["arms_crossed"]:
                confidence += 0.6
                feedback.append("✓ Arms crossed - beautiful!")
            else:
                feedback.append("✗ Cross arms over chest")
            
            if hand_features["left_detected"] and hand_features["right_detected"]:
                confidence += 0.3
                feedback.append("✓ Both hands on chest")
            
            confidence += 0.1
        
        elif sign_name == "FAMILY":
            if hand_features["left_detected"] and hand_features["right_detected"]:
                confidence += 0.4
                feedback.append("✓ Both hands forming F")
                
                avg_spread = (hand_features["left_spread"] + hand_features["right_spread"]) / 2
                if 0.06 < avg_spread < 0.10:
                    confidence += 0.4
                    feedback.append("✓ Good F-shape!")
                else:
                    feedback.append("✗ Adjust F-shape")
            else:
                feedback.append("✗ Use both hands")
            
            confidence += 0.2
        
        elif sign_name == "PLEASE":
            if 0.4 < hand_features["right_position_y"] < 0.7:
                confidence += 0.5
                feedback.append("✓ Hand on chest")
            else:
                feedback.append("✗ Move hand to chest")
            
            avg_spread = (hand_features["left_spread"] + hand_features["right_spread"]) / 2
            if avg_spread > 0.07:
                confidence += 0.3
                feedback.append("✓ Palm open")
            
            confidence += 0.2
        
        elif sign_name == "HELP":
            if hand_features["left_detected"] and hand_features["right_detected"]:
                confidence += 0.5
                feedback.append("✓ Both hands - good support!")
            else:
                feedback.append("✗ Show both hands")
            
            if 0.4 < hand_features["right_position_y"] < 0.7:
                confidence += 0.3
                feedback.append("✓ Good hand position")
            
            confidence += 0.2
        
        elif sign_name == "GOOD":
            if pose_features["right_arm_raised"] or pose_features["left_arm_raised"]:
                confidence += 0.5
                feedback.append("✓ Arm raised - thumbs up!")
            else:
                feedback.append("✗ Raise your arm")
            
            avg_spread = (hand_features["left_spread"] + hand_features["right_spread"]) / 2
            if avg_spread < 0.07:
                confidence += 0.3
                feedback.append("✓ Good fist shape")
            
            confidence += 0.2
        
        elif sign_name == "MORNING":
            if pose_features["right_arm_raised"] or pose_features["left_arm_raised"]:
                confidence += 0.5
                feedback.append("✓ Rising motion - like sun!")
            else:
                feedback.append("✗ Raise hand upward")
            
            confidence += 0.5
        
        elif sign_name == "FATHER":
            if hand_features["right_position_y"] < 0.4:
                confidence += 0.6
                feedback.append("✓ Hand at forehead area")
            else:
                feedback.append("✗ Touch forehead then chin")
            
            confidence += 0.4
        
        return {
            "detected": True,
            "confidence": min(confidence, 1.0),
            "feedback": " | ".join(feedback),
            "score": int(confidence * 100)
        }
    
    def analyze_phrase(self, results):
        """Analyze full phrase"""
        pose_features = self.extract_pose_features(results.pose_landmarks)
        hand_features = self.extract_hand_features(results.left_hand_landmarks, results.right_hand_landmarks)
        
        current_sign = self.get_current_sign()
        if not current_sign:
            return {
                "detected": True,
                "confidence": 1.0,
                "feedback": "✓ Phrase complete!",
                "score": 100,
                "sign_progress": 100
            }
        
        # Analyze current sign
        sign_feedback = self.analyze_sign(current_sign, pose_features, hand_features)
        
        # Add progress info
        phrase = self.get_current_phrase()
        total_signs = len(phrase["signs"])
        sign_feedback["sign_progress"] = (self.current_sign_index / total_signs) * 100
        sign_feedback["current_sign"] = current_sign
        sign_feedback["signs_completed"] = self.current_sign_index
        sign_feedback["total_signs"] = total_signs
        
        # Auto-advance if high confidence
        if sign_feedback["score"] >= 70 and self.recording:
            current_time = time.time()
            if current_sign not in self.sign_detected_time:
                self.sign_detected_time[current_sign] = current_time
            elif current_time - self.sign_detected_time[current_sign] > 1.0:  # Hold for 1 second
                if self.current_sign_index < total_signs - 1:
                    self.current_sign_index += 1
                    self.sign_detected_time = {}
                    print(f"  ✓ {current_sign} detected! Moving to next sign...")
        
        return sign_feedback
    
    def draw_phrase_panel(self, frame):
        """Draw phrase instruction panel"""
        h, w, _ = frame.shape
        
        phrase = self.get_current_phrase()
        if not phrase:
            return frame
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 260), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_offset = 35
        
        # Title
        title = f"Phrase {self.current_phrase_index + 1}/5: {phrase['phrase']}"
        cv2.putText(frame, title, (20, y_offset),
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)
        y_offset += 40
        
        # Signs sequence
        signs_text = " → ".join(phrase['signs'])
        cv2.putText(frame, f"Signs: {signs_text}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 35
        
        # Description
        cv2.putText(frame, phrase['description'], (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y_offset += 30
        
        # Tips
        cv2.putText(frame, f"Tips: {phrase['tips']}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 35
        
        # Current sign
        current_sign = self.get_current_sign()
        if current_sign:
            sign_text = f"Current: {current_sign} ({self.current_sign_index + 1}/{len(phrase['signs'])})"
            cv2.putText(frame, sign_text, (20, y_offset),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "✓ All signs completed!", (20, y_offset),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 40
        
        # Recording status
        if not self.recording:
            cv2.putText(frame, f"Press SPACE to start ({phrase['duration']}s)", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            elapsed = time.time() - self.recording_start_time
            remaining = max(0, phrase['duration'] - elapsed)
            cv2.putText(frame, f"RECORDING... {remaining:.1f}s", (20, y_offset),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
        
        return frame
    
    def draw_signing_area_guide(self, frame, results=None):
        """Draw visual guide for optimal signing area"""
        h, w, _ = frame.shape
        
        # Draw signing area rectangle (larger area for full-body)
        # Optimal signing area: center 70% of frame
        margin_x = int(w * 0.15)
        margin_y = int(h * 0.1)
        
        # Check if person is in frame
        in_frame = False
        position_status = "Position yourself in frame"
        status_color = (0, 165, 255)  # Orange
        
        if results and results.pose_landmarks:
            nose = results.pose_landmarks.landmark[0]
            nose_x = int(nose.x * w)
            nose_y = int(nose.y * h)
            
            # Check if person is within optimal area
            if margin_x < nose_x < (w - margin_x) and margin_y < nose_y < (h - margin_y):
                in_frame = True
                position_status = "✓ Perfect position!"
                status_color = (0, 255, 0)  # Green
            else:
                position_status = "Move into green area"
                status_color = (0, 165, 255)  # Orange
        
        # Draw guide rectangle with appropriate color
        guide_color = (0, 255, 0) if in_frame else (0, 165, 255)
        cv2.rectangle(frame, 
                     (margin_x, margin_y), 
                     (w - margin_x, h - margin_y),
                     guide_color, 3)
        
        # Add corner markers for better visibility
        corner_size = 40
        corners = [
            (margin_x, margin_y),  # Top-left
            (w - margin_x, margin_y),  # Top-right
            (margin_x, h - margin_y),  # Bottom-left
            (w - margin_x, h - margin_y)  # Bottom-right
        ]
        
        for x, y in corners:
            cv2.line(frame, (x - corner_size, y), (x + corner_size, y), guide_color, 4)
            cv2.line(frame, (x, y - corner_size), (x, y + corner_size), guide_color, 4)
        
        # Add position status
        cv2.putText(frame, position_status, 
                   (margin_x + 20, margin_y - 15),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, status_color, 2)
        
        # Add distance guide
        cv2.putText(frame, "Stand 1-2 meters from camera", 
                   (margin_x + 20, h - margin_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def draw_feedback_panel(self, frame, feedback):
        """Draw feedback panel"""
        h, w, _ = frame.shape
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 200), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        if feedback and feedback.get("detected"):
            score = feedback.get("score", 0)
            
            # Score bar
            bar_width = int((w - 40) * (score / 100))
            color = (0, 255, 0) if score >= 70 else (0, 255, 255) if score >= 40 else (0, 0, 255)
            
            cv2.rectangle(frame, (20, h - 180), (20 + bar_width, h - 160), color, -1)
            cv2.rectangle(frame, (20, h - 180), (w - 20, h - 160), (255, 255, 255), 2)
            
            # Score
            cv2.putText(frame, f"Score: {score}%", (20, h - 190),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            
            # Progress
            progress = feedback.get("sign_progress", 0)
            cv2.putText(frame, f"Phrase: {progress:.0f}%", (w - 200, h - 190),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Feedback
            feedback_text = feedback.get("feedback", "")
            y_offset = h - 130
            
            words = feedback_text.split(" | ")
            for word in words[:3]:
                cv2.putText(frame, word, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
        
        # Controls
        controls = "SPACE: Record | N: Next | R: Repeat | C: Next sign | Q: Quit"
        cv2.putText(frame, controls, (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        return frame
    
    def save_result(self, phrase, score, feedback):
        """Save practice result"""
        self.feedback_history.append({
            "phrase": phrase,
            "score": score,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        })
    
    def show_results(self):
        """Show final results"""
        print("\n" + "=" * 70)
        print("PHRASE PRACTICE SESSION COMPLETE!")
        print("=" * 70)
        
        if not self.feedback_history:
            print("No phrases practiced.")
            return
        
        total_score = 0
        print("\nYour Performance:\n")
        
        for i, result in enumerate(self.feedback_history, 1):
            score = result["score"]
            total_score += score
            
            indicator = "🌟 Excellent!" if score >= 70 else "👍 Good!" if score >= 50 else "📚 Keep practicing"
            
            print(f"{i}. {result['phrase']}: {score}% {indicator}")
            print(f"   {result['feedback']}\n")
        
        avg_score = total_score / len(self.feedback_history)
        print("=" * 70)
        print(f"AVERAGE SCORE: {avg_score:.1f}%")
        
        if avg_score >= 70:
            print("🎉 Outstanding! You're mastering GSL phrases!")
        elif avg_score >= 50:
            print("👏 Good progress! Keep practicing for fluency!")
        else:
            print("💪 Keep practicing! Phrases take time to master!")
        
        print("=" * 70)
    
    def run(self, camera_index=0):
        """Run phrase practice"""
        print("\n" + "=" * 70)
        print("GSL PHRASE PRACTICE - FULL-BODY RECOGNITION")
        print("=" * 70)
        print("\nPractice these 5 GSL phrases:\n")
        for i, phrase in enumerate(self.PRACTICE_PHRASES, 1):
            print(f"{i}. {phrase['phrase']}")
            print(f"   Signs: {' → '.join(phrase['signs'])}")
            print(f"   Duration: {phrase['duration']} seconds\n")
        
        print("Controls:")
        print("  SPACE - Start recording phrase")
        print("  N - Next phrase")
        print("  R - Repeat current phrase")
        print("  C - Skip to next sign")
        print("  Q - Quit\n")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("✗ Cannot open webcam")
            return
        
        # Set larger resolution for better full-body capture
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        # Get actual resolution
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {actual_width}x{actual_height}")
        
        print("✓ Camera ready! Show your phrases!\n")
        
        current_feedback = None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
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
                
                # Draw signing area guide (when not recording for clarity)
                if not self.recording:
                    frame = self.draw_signing_area_guide(frame, results)
                
                # Analyze if recording
                if self.recording:
                    current_feedback = self.analyze_phrase(results)
                
                # Check timeout
                phrase = self.get_current_phrase()
                if self.recording and phrase:
                    elapsed = time.time() - self.recording_start_time
                    if elapsed >= phrase['duration']:
                        self.recording = False
                        if current_feedback:
                            self.save_result(
                                phrase['phrase'],
                                current_feedback.get('score', 0),
                                current_feedback.get('feedback', '')
                            )
                            print(f"✓ {phrase['phrase']}: {current_feedback.get('score', 0)}%")
                
                # Draw UI
                frame = self.draw_phrase_panel(frame)
                frame = self.draw_feedback_panel(frame, current_feedback)
                
                cv2.imshow('GSL Phrase Practice - Full Body', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' ') and not self.recording:
                    self.recording = True
                    self.recording_start_time = time.time()
                    self.phrase_start_time = time.time()
                    self.current_sign_index = 0
                    self.sign_detected_time = {}
                    current_feedback = None
                elif key == ord('n'):
                    if self.current_phrase_index < len(self.PRACTICE_PHRASES) - 1:
                        self.current_phrase_index += 1
                        self.current_sign_index = 0
                        self.sign_detected_time = {}
                        current_feedback = None
                        print(f"\nNext phrase: {self.get_current_phrase()['phrase']}")
                    else:
                        print("\nAll phrases completed! Press Q for results.")
                elif key == ord('r'):
                    self.current_sign_index = 0
                    self.sign_detected_time = {}
                    current_feedback = None
                    print(f"\nRepeating: {self.get_current_phrase()['phrase']}")
                elif key == ord('c'):
                    phrase = self.get_current_phrase()
                    if self.current_sign_index < len(phrase['signs']) - 1:
                        self.current_sign_index += 1
                        self.sign_detected_time = {}
                        print(f"  Next sign: {self.get_current_sign()}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.holistic.close()
            self.show_results()


def main():
    """Main function"""
    session = GSLPhrasePracticeFullBody()
    session.run(camera_index=0)


if __name__ == "__main__":
    main()
