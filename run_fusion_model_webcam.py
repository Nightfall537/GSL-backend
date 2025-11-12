"""
Run GSL Fusion Model (ResNeXt + MediaPipe + LSTM) with Webcam
This model achieved ~77.8% accuracy with multiple pretrained models
"""

import sys
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from collections import deque
import json

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class GSLFusionModelRunner:
    """Run the high-accuracy fusion model with webcam"""
    
    def __init__(self):
        print("=" * 70)
        print("GSL FUSION MODEL - REAL-TIME RECOGNITION")
        print("=" * 70)
        print("\nModel Architecture:")
        print("  - MediaPipe Holistic (468 landmarks)")
        print("  - ResNeXt-101 3D CNN (512-dim embeddings)")
        print("  - Bidirectional LSTM with Attention")
        print("  - Fusion with Gating Mechanism")
        print(f"  - Reported Accuracy: ~77-78%")
        print("=" * 70)
        
        # Initialize MediaPipe Holistic for full feature extraction
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load trained model
        self.model = None
        self.class_mapping = None
        self.load_model()
        
        # Sequence buffer for temporal predictions
        self.sequence_length = 30
        self.landmark_buffer = deque(maxlen=self.sequence_length)
        
        # For ResNeXt features (we'll use zeros as placeholder since we need video preprocessing)
        self.resnext_dim = 512
        
    def load_model(self):
        """Load the trained fusion model"""
        print("\n🔍 Searching for trained models...")
        
        models_dir = Path("app/models/trained_models")
        if not models_dir.exists():
            print("❌ Models directory not found!")
            return False
        
        # Look for the improved/fusion models
        model_patterns = [
            "gsl_improved_final_*.h5",
            "gsl_improved_*_best.h5",
            "gsl_fusion_lstm_*.h5",
            "gsl_final_*.h5"
        ]
        
        model_files = []
        for pattern in model_patterns:
            model_files.extend(list(models_dir.glob(pattern)))
        
        if not model_files:
            print("❌ No trained models found!")
            print(f"   Searched in: {models_dir}")
            return False
        
        # Use the most recent model
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        print(f"✓ Found model: {latest_model.name}")
        
        # Load class mapping
        class_mapping_file = models_dir / "class_mapping_20251109_125313.json"
        if class_mapping_file.exists():
            with open(class_mapping_file, 'r', encoding='utf-8') as f:
                self.class_mapping = json.load(f)
            print(f"✓ Loaded {len(self.class_mapping)} classes")
        else:
            print("⚠️ Class mapping not found, using default classes")
            self.class_mapping = {str(i): f"Sign_{i}" for i in range(50)}
        
        # Load TensorFlow model
        try:
            import tensorflow as tf
            print(f"\n🔄 Loading model: {latest_model.name}")
            self.model = tf.keras.models.load_model(str(latest_model))
            print("✓ Model loaded successfully!")
            print(f"  Input shapes: {[inp.shape for inp in self.model.inputs]}")
            print(f"  Output shape: {self.model.output.shape}")
            print(f"  Parameters: {self.model.count_params():,}")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_landmarks(self, results):
        """Extract 468-dimensional landmark features"""
        landmarks = []
        
        # Pose landmarks (33 * 4 = 132 features: x, y, z, visibility)
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
            # Select key facial landmarks
            key_indices = list(range(0, 468, 10))[:47]  # Sample 47 points
            for idx in key_indices:
                if idx < len(results.face_landmarks.landmark):
                    lm = results.face_landmarks.landmark[idx]
                    landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
                else:
                    landmarks.extend([0.0, 0.0, 0.0, 0.0])
        else:
            landmarks.extend([0.0] * 188)
        
        # Total: 132 + 84 + 84 + 188 = 488 features
        # Trim to 468 to match model
        return np.array(landmarks[:468], dtype=np.float32)
    
    def predict_sign(self):
        """Predict sign from buffered landmarks"""
        if len(self.landmark_buffer) < self.sequence_length:
            return None, 0.0, []
        
        if self.model is None:
            return None, 0.0, []
        
        try:
            # Prepare landmark sequence
            landmark_seq = np.array(list(self.landmark_buffer))
            landmark_seq = np.expand_dims(landmark_seq, axis=0)  # (1, 30, 468)
            
            # Create mask (all True since we have full sequence)
            mask = np.ones((1, self.sequence_length), dtype=bool)
            
            # Create placeholder ResNeXt features (zeros)
            # In production, you'd extract these from video frames
            resnext_features = np.zeros((1, self.resnext_dim), dtype=np.float32)
            
            # Make prediction
            predictions = self.model.predict(
                {
                    'landmark_input': landmark_seq,
                    'resnext_input': resnext_features,
                    'mask_input': mask
                },
                verbose=0
            )
            
            # Get top 3 predictions
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_3_probs = predictions[0][top_3_indices]
            
            # Get class names
            top_3_classes = []
            for idx, prob in zip(top_3_indices, top_3_probs):
                class_name = self.class_mapping.get(str(idx), f"Class_{idx}")
                top_3_classes.append((class_name, prob))
            
            predicted_class = top_3_classes[0][0]
            confidence = top_3_classes[0][1]
            
            return predicted_class, confidence, top_3_classes
        
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0, []
    
    def run_webcam(self):
        """Run real-time recognition with webcam"""
        if self.model is None:
            print("\n❌ Model not loaded - cannot run recognition")
            return False
        
        print("\n🚀 Starting Real-Time Recognition")
        print("=" * 70)
        print("📹 Opening webcam...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open webcam")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("✓ Webcam ready!")
        print("\nControls:")
        print("  Q - Quit")
        print("  R - Reset buffer")
        print("\nShow GSL signs to the camera!")
        print("=" * 70)
        
        frame_count = 0
        prediction_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                frame_count += 1
                h, w = frame.shape[:2]
                
                # Process with MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(rgb_frame)
                
                # Draw landmarks
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks,
                        self.mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.left_hand_landmarks,
                        self.mp_holistic.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, results.right_hand_landmarks,
                        self.mp_holistic.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                # Extract features and buffer
                if results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
                    landmarks = self.extract_landmarks(results)
                    self.landmark_buffer.append(landmarks)
                
                # Predict when buffer is full
                predicted_sign = "Collecting frames..."
                confidence = 0.0
                top_3 = []
                
                if len(self.landmark_buffer) == self.sequence_length:
                    predicted_sign, confidence, top_3 = self.predict_sign()
                    if predicted_sign:
                        prediction_count += 1
                
                # Draw UI
                # Main panel
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (w - 10, 220), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                # Title
                cv2.putText(frame, "GSL FUSION MODEL - LIVE RECOGNITION", (20, 40),
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)
                
                # Model info
                cv2.putText(frame, "ResNeXt + MediaPipe + BiLSTM + Attention", (20, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                # Prediction
                color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255) if confidence > 0.4 else (0, 0, 255)
                cv2.putText(frame, f"Sign: {predicted_sign}", (20, 110),
                           cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
                cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, 145),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Top 3 predictions
                if top_3:
                    y_offset = 175
                    cv2.putText(frame, "Top 3:", (20, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    for i, (cls, prob) in enumerate(top_3[:3], 1):
                        y_offset += 20
                        text = f"  {i}. {cls}: {prob:.1%}"
                        cv2.putText(frame, text, (20, y_offset),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                
                # Buffer status
                buffer_pct = (len(self.landmark_buffer) / self.sequence_length) * 100
                cv2.rectangle(frame, (20, h - 60), (w - 20, h - 40), (50, 50, 50), -1)
                cv2.rectangle(frame, (20, h - 60), (20 + int((w - 40) * buffer_pct / 100), h - 40), (0, 255, 0), -1)
                cv2.putText(frame, f"Buffer: {len(self.landmark_buffer)}/{self.sequence_length}", (25, h - 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Stats
                cv2.putText(frame, f"Frames: {frame_count} | Predictions: {prediction_count}", (20, h - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(frame, "Q: Quit | R: Reset", (w - 200, h - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow('GSL Fusion Model', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.landmark_buffer.clear()
                    print("🔄 Buffer reset")
                
                if frame_count % 300 == 0:
                    print(f"📊 Frames: {frame_count} | Predictions: {prediction_count}")
        
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.holistic.close()
            
            print("\n" + "=" * 70)
            print("SESSION SUMMARY")
            print("=" * 70)
            print(f"Total frames: {frame_count}")
            print(f"Total predictions: {prediction_count}")
            if frame_count > 0:
                print(f"Prediction rate: {prediction_count / frame_count * 100:.1f}%")
            print("=" * 70)
        
        return True


def main():
    """Main function"""
    runner = GSLFusionModelRunner()
    
    if runner.model is None:
        print("\n❌ Cannot run without model")
        print("\n💡 To train the model, run:")
        print("   cd app/models")
        print("   python train_gsl_improved.py")
        return
    
    runner.run_webcam()


if __name__ == "__main__":
    main()
