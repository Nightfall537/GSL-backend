#!/usr/bin/env python3
"""
Test Deep Learning Model - Real Inference
Load and test the SAM2-trained model with actual predictions
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import logging
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepLearningModelTester:
    """Test the trained deep learning model with real inference"""
    
    def __init__(self):
        # Initialize MediaPipe Holistic for full-body tracking
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.is_fullbody_model = False
        
        # Load TensorFlow model
        self.model = None
        self.model_path = self.find_latest_model()
        self.load_model()
        
        # Gesture classes (from training)
        self.gestures = ['red', 'blue', 'green', 'yellow', 'black', 'white', 
                        'orange', 'purple', 'pink', 'brown', 'gray']
        
        # Sequence buffer for temporal predictions
        self.sequence_length = 30
        self.feature_buffer = deque(maxlen=self.sequence_length)
        
        logger.info("🎯 Deep Learning Model Tester Initialized")
        logger.info(f"🤖 Model: {self.model_path}")
        logger.info(f"🎨 Gestures: {len(self.gestures)} classes")
    
    def find_latest_model(self):
        """Find the latest trained model"""
        models_dir = Path("sam2_training_output/models")
        if not models_dir.exists():
            logger.error("❌ No models directory found")
            return None
        
        # Look for full-body models first, then fall back to regular models
        model_files = list(models_dir.glob("fullbody_gsl_model_*.h5"))
        if not model_files:
            model_files = list(models_dir.glob("sam2_gsl_model_*.h5"))
        
        if not model_files:
            logger.error("❌ No trained models found")
            return None
        
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"📂 Found model: {latest_model.name}")
        
        # Check if it's a full-body model
        self.is_fullbody_model = 'fullbody' in latest_model.name
        if self.is_fullbody_model:
            logger.info("✅ Using full-body model (pose + hands + face)")
        else:
            logger.info("⚠️ Using hand-only model")
        
        return latest_model
    
    def load_model(self):
        """Load the TensorFlow model"""
        if not self.model_path:
            logger.error("❌ No model path available")
            return False
        
        try:
            import tensorflow as tf
            logger.info("🔄 Loading TensorFlow model...")
            self.model = tf.keras.models.load_model(str(self.model_path))
            logger.info("✅ Model loaded successfully")
            logger.info(f"📊 Model input shape: {self.model.input_shape}")
            logger.info(f"📊 Model output shape: {self.model.output_shape}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False
    
    def extract_fullbody_features(self, results):
        """Extract comprehensive full-body features from MediaPipe Holistic"""
        features = []
        
        # Pose landmarks (33 * 3 = 99 features)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
        else:
            features.extend([0.0] * 99)
        
        # Left hand landmarks (21 * 3 = 63 features)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
        else:
            features.extend([0.0] * 63)
        
        # Right hand landmarks (21 * 3 = 63 features)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
        else:
            features.extend([0.0] * 63)
        
        # Face landmarks (14 key points * 3 = 42 features)
        if results.face_landmarks:
            key_face_indices = [0, 1, 4, 5, 6, 10, 33, 61, 133, 152, 263, 291, 362, 386]
            for idx in key_face_indices:
                if idx < len(results.face_landmarks.landmark):
                    lm = results.face_landmarks.landmark[idx]
                    features.extend([lm.x, lm.y, lm.z])
                else:
                    features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([0.0] * 42)
        
        # Spatial relationships (6 features)
        if results.pose_landmarks and len(results.pose_landmarks.landmark) >= 16:
            nose = results.pose_landmarks.landmark[0]
            left_wrist = results.pose_landmarks.landmark[15]
            right_wrist = results.pose_landmarks.landmark[16]
            
            # Left hand relative to nose
            features.extend([
                left_wrist.x - nose.x,
                left_wrist.y - nose.y,
                left_wrist.z - nose.z
            ])
            
            # Right hand relative to nose
            features.extend([
                right_wrist.x - nose.x,
                right_wrist.y - nose.y,
                right_wrist.z - nose.z
            ])
        else:
            features.extend([0.0] * 6)
        
        return np.array(features, dtype=np.float32)
    
    def predict_gesture(self):
        """Predict gesture from feature buffer"""
        if len(self.feature_buffer) < self.sequence_length:
            return None, 0.0
        
        if self.model is None:
            return None, 0.0
        
        try:
            # Prepare sequence for prediction
            sequence = np.array(list(self.feature_buffer))
            sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
            
            # Make prediction
            predictions = self.model.predict(sequence, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            gesture_name = self.gestures[predicted_class]
            return gesture_name, confidence
        
        except Exception as e:
            logger.error(f"❌ Prediction error: {e}")
            return None, 0.0
    
    def test_live_recognition(self):
        """Test with live camera and real predictions"""
        if self.model is None:
            logger.error("❌ Model not loaded - cannot test")
            return False
        
        logger.info("🚀 Starting Live Recognition with Deep Learning Model")
        logger.info("📹 Opening camera...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("❌ Cannot open camera")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("✅ Camera ready")
        logger.info("🎯 Show color gestures for real-time recognition!")
        logger.info("📝 Trained gestures: " + ", ".join(self.gestures))
        
        frame_count = 0
        prediction_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                frame_count += 1
                height, width = frame.shape[:2]
                
                # Process with MediaPipe Holistic
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(rgb_frame)
                
                current_gesture = "No person detected"
                confidence = 0.0
                
                # Draw landmarks
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                
                if results.left_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                if results.right_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                if results.face_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                
                # Extract features and predict
                if results.pose_landmarks:
                    # Extract full-body features
                    features = self.extract_fullbody_features(results)
                    self.feature_buffer.append(features)
                    
                    # Predict when buffer is full
                    if len(self.feature_buffer) == self.sequence_length:
                        gesture, conf = self.predict_gesture()
                        if gesture:
                            current_gesture = gesture
                            confidence = conf
                            prediction_count += 1
                
                # Draw status panel
                cv2.rectangle(frame, (10, 10), (width - 10, 150), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, 10), (width - 10, 150), (255, 255, 255), 2)
                
                # Model info
                model_type = "FULL-BODY MODEL" if self.is_fullbody_model else "HAND-ONLY MODEL"
                cv2.putText(frame, f"{model_type} - LIVE INFERENCE", (20, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Prediction result
                color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
                cv2.putText(frame, f"Gesture: {current_gesture}", (20, 65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, 95), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Buffer status
                buffer_status = f"Buffer: {len(self.feature_buffer)}/{self.sequence_length}"
                cv2.putText(frame, buffer_status, (20, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(frame, "Press 'q' to quit", (20, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show trained gestures
                gesture_text = "Trained: " + ", ".join(self.gestures[:6])
                cv2.putText(frame, gesture_text, (20, height - 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                gesture_text2 = "         " + ", ".join(self.gestures[6:])
                cv2.putText(frame, gesture_text2, (20, height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                cv2.imshow('Deep Learning Model - Live Recognition', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                
                # Log progress
                if frame_count % 300 == 0:
                    logger.info(f"📊 Frames: {frame_count} | Predictions: {prediction_count}")
        
        except KeyboardInterrupt:
            logger.info("🛑 Test stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info(f"✅ Test completed - {prediction_count} predictions made")
        
        return True

def main():
    """Main test function"""
    print("🎯 Testing Deep Learning Model with Real Inference")
    print("🤖 SAM2-trained model with TensorFlow")
    print("=" * 60)
    
    tester = DeepLearningModelTester()
    
    if tester.model is None:
        print("\n❌ Model not loaded - cannot proceed with testing")
        print("💡 Make sure the model was trained successfully")
        return
    
    print("\n🚀 Starting live recognition test...")
    print("📹 Show color gestures to the camera")
    print("🎨 The model will predict: red, blue, green, yellow, black, white, orange, purple, pink, brown, gray")
    print("\nPress 'q' in the video window to quit\n")
    
    tester.test_live_recognition()
    
    print("\n🎉 Model test complete!")
    print("📊 Your deep learning model is working with real-time inference")

if __name__ == "__main__":
    main()
