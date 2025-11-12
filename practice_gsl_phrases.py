"""
GSL Phrase Practice Session

Practice signing complete phrases in Ghanaian Sign Language.
Get real-time feedback on your signing performance.
"""

import cv2
import mediapipe as mp
import time
import numpy as np
from datetime import datetime
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class GSLPhrasePractice:
    """Practice GSL phrases with real-time feedback"""
    
    # 5 common GSL phrases to practice
    PRACTICE_PHRASES = [
        {
            "phrase": "HELLO, MY NAME",
            "signs": ["HELLO", "MY", "NAME"],
            "description": "Wave hand → Point to chest → Touch fingertips to chest",
            "tips": "Smooth transitions between signs, maintain eye contact",
            "difficulty": "Easy"
        },
        {
            "phrase": "THANK YOU FRIEND",
            "signs": ["THANK YOU", "FRIEND"],
            "description": "Fingers to chin moving forward → Hook index fingers together",
            "tips": "Express gratitude with facial expression, gentle linking motion",
            "difficulty": "Easy"
        },
        {
            "phrase": "I LOVE MY FAMILY",
            "signs": ["ME", "LOVE", "MY", "FAMILY"],
            "description": "Point to self → Cross arms on chest → Point to chest → F-shape circle",
            "tips": "Show emotion, smooth flow between signs",
            "difficulty": "Medium"
        },
        {
            "phrase": "PLEASE HELP ME",
            "signs": ["PLEASE", "HELP", "ME"],
            "description": "Circular motion on chest → One hand supports other → Point to self",
            "tips": "Polite facial expression, clear hand positioning",
            "difficulty": "Medium"
        },
        {
            "phrase": "GOOD MORNING FATHER",
            "signs": ["GOOD", "MORNING", "FATHER"],
            "description": "Thumbs up → Sun rising gesture → Touch forehead then chin",
            "tips": "Respectful demeanor, clear morning gesture",
            "difficulty": "Medium"
        }
    ]
    
    def __init__(self):
        """Initialize practice session"""
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.current_phrase_index = 0
        self.recording = False
        self.recording_start_time = None
        self.recording_duration = 5  # 5 seconds for phrases
        self.feedback_history = []
        
        # Create results directory
        self.results_dir = "phrase_practice_results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        # Tracking for phrase completion
        self.signs_detected = []
        self.current_sign_index = 0
    
    def get_current_phrase(self):
        """Get current phrase to practice"""
        if self.current_phrase_index < len(self.PRACTICE_PHRASES):
            return self.PRACTICE_PHRASES[self.current_phrase_index]
        return None
    
    def analyze_phrase_gesture(self, hand_landmarks, handedness):
        """Analyze gesture for phrase recognition"""
        if not hand_landmarks:
            return {
                "detected": False,
                "confidence": 0.0,
                "feedback": "No hands detected",
                "sign_progress": 0
            }
        
        current_phrase = self.get_current_phrase()
        if not current_phrase:
            return {"detected": False, "confidence": 0.0, "feedback": "Practice complete!"}
        
        # Get landmark positions
        landmarks = hand_landmarks.landmark
        
        # Calculate features
        fingers_spread = self.calculate_finger_spread(landmarks)
        hand_height = abs(landmarks[8].y - landmarks[0].y)
        hand_position = landmarks[9].y
        hand_center_x = landmarks[9].x
        
        # Analyze based on expected signs in phrase
        total_signs = len(current_phrase["signs"])
        expected_sign = current_phrase["signs"][self.current_sign_index] if self.current_sign_index < total_signs else None
        
        feedback = self.generate_phrase_feedback(
            expected_sign,
            fingers_spread,
            hand_height,
            hand_position,
            hand_center_x,
            handedness
        )
        
        # Track sign progress
        feedback["sign_progress"] = (self.current_sign_index / total_signs) * 100
        feedback["current_sign"] = expected_sign
        feedback["signs_completed"] = self.current_sign_index
        feedback["total_signs"] = total_signs
        
        return feedback
    
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
    
    def generate_phrase_feedback(self, expected_sign, fingers_spread, hand_height, 
                                 hand_position, hand_center_x, handedness):
        """Generate feedback for phrase signing"""
        confidence = 0.0
        feedback_text = []
        
        if not expected_sign:
            return {
                "detected": True,
                "confidence": 1.0,
                "feedback": "✓ Phrase complete!",
                "score": 100
            }
        
        # Analyze based on expected sign
        if expected_sign == "HELLO":
            if fingers_spread > 0.08:
                confidence += 0.5
                feedback_text.append("✓ Open palm")
            else:
                feedback_text.append("✗ Spread fingers")
            
            if hand_position < 0.5:
                confidence += 0.3
                feedback_text.append("✓ Hand raised")
            
            confidence += 0.2
            
        elif expected_sign == "MY" or expected_sign == "ME":
            if 0.4 < hand_center_x < 0.6:
                confidence += 0.5
                feedback_text.append("✓ Pointing to self")
            else:
                feedback_text.append("✗ Point to chest")
            
            confidence += 0.3
            
        elif expected_sign == "NAME":
            if 0.4 < hand_position < 0.6:
                confidence += 0.5
                feedback_text.append("✓ Hand at chest level")
            
            confidence += 0.3
            
        elif expected_sign == "THANK YOU":
            if hand_position < 0.4:
                confidence += 0.5
                feedback_text.append("✓ Hand near face")
            else:
                feedback_text.append("✗ Move to chin area")
            
            confidence += 0.3
            
        elif expected_sign == "FRIEND":
            if 0.05 < fingers_spread < 0.09:
                confidence += 0.5
                feedback_text.append("✓ Finger position good")
            
            confidence += 0.3
            
        elif expected_sign == "LOVE":
            if 0.4 < hand_position < 0.7:
                confidence += 0.5
                feedback_text.append("✓ Hand on chest")
            
            confidence += 0.3
            
        elif expected_sign == "FAMILY":
            if 0.06 < fingers_spread < 0.10:
                confidence += 0.5
                feedback_text.append("✓ F-shape forming")
            
            confidence += 0.3
            
        elif expected_sign == "PLEASE":
            if 0.4 < hand_position < 0.7:
                confidence += 0.4
                feedback_text.append("✓ Chest area")
            
            if fingers_spread > 0.07:
                confidence += 0.3
                feedback_text.append("✓ Palm open")
            
            confidence += 0.2
            
        elif expected_sign == "HELP":
            if 0.4 < hand_position < 0.7:
                confidence += 0.5
                feedback_text.append("✓ Good position")
            
            confidence += 0.3
            
        elif expected_sign == "GOOD":
            if fingers_spread < 0.07:
                confidence += 0.5
                feedback_text.append("✓ Thumbs up shape")
            
            confidence += 0.3
            
        elif expected_sign == "MORNING":
            if hand_position < 0.5:
                confidence += 0.5
                feedback_text.append("✓ Rising motion")
            
            confidence += 0.3
            
        elif expected_sign == "FATHER":
            if hand_position < 0.4:
                confidence += 0.5
                feedback_text.append("✓ Head area")
            
            confidence += 0.3
        
        # Add hand info
        hand_label = handedness.classification[0].label if handedness else "Unknown"
        feedback_text.append(f"Using {hand_label} hand")
        
        # Check if sign is complete (high confidence)
        if confidence > 0.6:
            feedback_text.append(f"✓ {expected_sign} detected!")
        
        return {
            "detected": True,
            "confidence": min(confidence, 1.0),
            "feedback": " | ".join(feedback_text),
            "score": int(confidence * 100)
        }
    
    def draw_phrase_panel(self, frame):
        """Draw phrase instruction panel"""
        h, w, _ = frame.shape
        
        current_phrase = self.get_current_phrase()
        if not current_phrase:
            return frame
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 250), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y_offset = 40
        
        # Phrase number and text
        phrase_text = f"Phrase {self.current_phrase_index + 1}/5: {current_phrase['phrase']}"
        cv2.putText(frame, phrase_text, (20, y_offset),
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
        y_offset += 40
        
        # Individual signs
        signs_text = "Signs: " + " → ".join(current_phrase['signs'])
        cv2.putText(frame, signs_text, (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 35
        
        # Description
        cv2.putText(frame, current_phrase['description'], (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 30
        
        # Tips
        cv2.putText(frame, f"Tips: {current_phrase['tips']}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 35
        
        # Current sign progress
        if self.current_sign_index < len(current_phrase['signs']):
            current_sign = current_phrase['signs'][self.current_sign_index]
            cv2.putText(frame, f"Current Sign: {current_sign} ({self.current_sign_index + 1}/{len(current_phrase['signs'])})",
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "✓ All signs completed!", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 35
        
        # Recording status
        if not self.recording:
            cv2.putText(frame, "Press SPACE to start recording (5 seconds)", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            elapsed = time.time() - self.recording_start_time
            remaining = max(0, self.recording_duration - elapsed)
            cv2.putText(frame, f"RECORDING... {remaining:.1f}s", (20, y_offset),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2)
        
        return frame
    
    def draw_feedback_panel(self, frame, feedback):
        """Draw feedback panel"""
        h, w, _ = frame.shape
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h - 180), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        if feedback and feedback.get("detected"):
            score = feedback.get("score", 0)
            
            # Score bar
            bar_width = int((w - 40) * (score / 100))
            color = (0, 255, 0) if score >= 70 else (0, 255, 255) if score >= 40 else (0, 0, 255)
            
            cv2.rectangle(frame, (20, h - 160), (20 + bar_width, h - 140), color, -1)
            cv2.rectangle(frame, (20, h - 160), (w - 20, h - 140), (255, 255, 255), 2)
            
            # Score text
            cv2.putText(frame, f"Score: {score}%", (20, h - 170),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 2)
            
            # Sign progress
            sign_progress = feedback.get("sign_progress", 0)
            progress_text = f"Phrase Progress: {sign_progress:.0f}%"
            cv2.putText(frame, progress_text, (w - 300, h - 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Feedback text
            feedback_text = feedback.get("feedback", "")
            y_offset = h - 110
            
            # Split feedback into lines
            words = feedback_text.split(" | ")
            for word in words[:3]:  # Show max 3 lines
                cv2.putText(frame, word, (20, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
        
        return frame
    
    def draw_controls(self, frame):
        """Draw control instructions"""
        h, w, _ = frame.shape
        
        controls = [
            "SPACE: Record phrase (5s)",
            "N: Next phrase",
            "R: Repeat phrase",
            "C: Continue to next sign",
            "Q: Quit"
        ]
        
        x_offset = w - 280
        y_offset = h - 150
        
        for control in controls:
            cv2.putText(frame, control, (x_offset, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y_offset += 22
    
    def save_phrase_result(self, phrase, score, feedback):
        """Save practice result"""
        result = {
            "phrase": phrase,
            "score": score,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }
        self.feedback_history.append(result)
    
    def show_final_results(self):
        """Display final results"""
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
            
            if score >= 70:
                indicator = "🌟 Excellent!"
            elif score >= 50:
                indicator = "👍 Good!"
            elif score >= 30:
                indicator = "📚 Keep practicing"
            else:
                indicator = "💪 Needs work"
            
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
        """Run phrase practice session"""
        print("\n" + "=" * 70)
        print("GSL PHRASE PRACTICE SESSION")
        print("=" * 70)
        print("\nYou will practice 5 GSL phrases:")
        for i, phrase in enumerate(self.PRACTICE_PHRASES, 1):
            print(f"{i}. {phrase['phrase']}")
            print(f"   Signs: {' → '.join(phrase['signs'])}")
        
        print("\nControls:")
        print("  SPACE - Record your phrase (5 seconds)")
        print("  N - Next phrase")
        print("  R - Repeat current phrase")
        print("  C - Continue to next sign in phrase")
        print("  Q - Quit")
        print("\nStarting camera...")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("✗ Error: Could not open webcam")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("✓ Camera ready! Show your phrases!\n")
        
        current_feedback = None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    print("Failed to read frame, retrying...")
                    continue
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                
                # Draw hand landmarks
                if results.multi_hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                        
                        # Analyze if recording
                        if self.recording and results.multi_handedness:
                            handedness = results.multi_handedness[hand_idx]
                            current_feedback = self.analyze_phrase_gesture(
                                hand_landmarks, handedness
                            )
                
                # Check recording timeout
                if self.recording:
                    elapsed = time.time() - self.recording_start_time
                    if elapsed >= self.recording_duration:
                        self.recording = False
                        if current_feedback:
                            current_phrase = self.get_current_phrase()
                            self.save_phrase_result(
                                current_phrase["phrase"],
                                current_feedback.get("score", 0),
                                current_feedback.get("feedback", "")
                            )
                            print(f"✓ {current_phrase['phrase']}: {current_feedback.get('score', 0)}%")
                
                # Draw UI
                frame = self.draw_phrase_panel(frame)
                frame = self.draw_feedback_panel(frame, current_feedback)
                frame = self.draw_controls(frame)
                
                cv2.imshow('GSL Phrase Practice', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' ') and not self.recording:
                    self.recording = True
                    self.recording_start_time = time.time()
                    self.current_sign_index = 0
                    current_feedback = None
                elif key == ord('n'):
                    if self.current_phrase_index < len(self.PRACTICE_PHRASES) - 1:
                        self.current_phrase_index += 1
                        self.current_sign_index = 0
                        current_feedback = None
                        print(f"\nNext phrase: {self.get_current_phrase()['phrase']}")
                    else:
                        print("\nAll phrases completed! Press Q to see results.")
                elif key == ord('r'):
                    self.current_sign_index = 0
                    current_feedback = None
                    print(f"\nRepeating: {self.get_current_phrase()['phrase']}")
                elif key == ord('c'):
                    current_phrase = self.get_current_phrase()
                    if self.current_sign_index < len(current_phrase['signs']) - 1:
                        self.current_sign_index += 1
                        print(f"Next sign: {current_phrase['signs'][self.current_sign_index]}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            self.show_final_results()


def main():
    """Main function"""
    session = GSLPhrasePractice()
    session.run(camera_index=0)


if __name__ == "__main__":
    main()
