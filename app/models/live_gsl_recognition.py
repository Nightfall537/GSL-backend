#!/usr/bin/env python3
"""
Live GSL Recognition System
Real-time gesture recognition using webcam with trained fusion model
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import json
import mediapipe as mp
import tensorflow as tf
from pathlib import Path
from collections import deque
import time

print("🚀 GSL Live Recognition System")
print("=" * 80)

# Load the trained model
MODEL_PATH = "trained_models/gsl_final_20251109_125313_best.h5"
CLASS_MAPPING_PATH = "trained_models/class_mapping_20251109_125313.json"

print(f"📂 Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# Load class mapping
print(f"📂 Loading class mapping: {CLASS_MAPPING_PATH}")
with open(CLASS_MAPPING_PATH, 'r') as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(class_to_idx)
print(f"✅ Loaded {num_classes} gesture classes")
print(f"📋 Gestures: {', '.join(sorted(class_to_idx.keys()))}")

# Initialize MediaPipe Holistic
print("\n🔧 Initializing MediaPipe Holistic...")
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
print("✅ MediaPipe initialized")

# Initialize ResNeXt (for future use - currently using zero embeddings for live demo)
print("⚠️  Note: ResNeXt features require video buffer - using simplified mode")

# Gesture buffer
BUFFER_SIZE = 60  # Buffer 60 frames (~2 seconds at 30fps)
landmark_buffer = deque(maxlen=BUFFER_SIZE)
CONFIDENCE_THRESHOLD = 0.6

def extract_landmarks(results):
    """Extract 468 landmarks from MediaPipe results"""
    features = []
    
    # Pose (132)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        features.extend([0.0] * 132)
    
    # Left hand (63)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 63)
    
    # Right hand (63)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0] * 63)
    
    # Face (210 - 70 key points)
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
    
    # Ensure exact size
    if len(features) < 468:
        features.extend([0.0] * (468 - len(features)))
    elif len(features) > 468:
        features = features[:468]
    
    return np.array(features, dtype=np.float32)


def predict_gesture(landmark_sequence):
    """Predict gesture from landmark sequence"""
    if len(landmark_sequence) < 10:  # Need minimum frames
        return None, 0.0
    
    # Prepare input
    landmarks = np.array(landmark_sequence)[np.newaxis, :, :]  # [1, seq_len, 468]
    resnext_embedding = np.zeros((1, 512), dtype=np.float32)  # Placeholder
    mask = np.ones((1, len(landmark_sequence)), dtype=np.bool_)
    
    # Predict
    try:
        prediction = model.predict({
            'landmark_input': landmarks,
            'resnext_input': resnext_embedding,
            'mask_input': mask
        }, verbose=0)
        
        predicted_idx = np.argmax(prediction[0])
        confidence = prediction[0][predicted_idx]
        predicted_class = idx_to_class[predicted_idx]
        
        return predicted_class, confidence
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0.0


def draw_landmarks(image, results):
    """Draw landmarks on image"""
    # Draw pose
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
    
    # Draw hands
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )


def main():
    """Main live recognition loop"""
    print("\n" + "=" * 80)
    print("🎥 Starting webcam...")
    print("=" * 80)
    print("\n📋 Instructions:")
    print("   - Perform gestures in front of the camera")
    print("   - Hold gesture for 1-2 seconds for recognition")
    print("   - Press 'q' to quit")
    print("   - Press 'r' to reset buffer")
    print("\n" + "=" * 80)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    print("✅ Webcam opened successfully")
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    current_prediction = "Waiting..."
    current_confidence = 0.0
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = holistic.process(rgb_frame)
            
            # Extract landmarks
            if results.pose_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:
                landmarks = extract_landmarks(results)
                landmark_buffer.append(landmarks)
                
                # Predict every 15 frames (twice per second at 30fps)
                if len(landmark_buffer) >= 30 and frame_count % 15 == 0:
                    predicted_class, confidence = predict_gesture(list(landmark_buffer))
                    
                    if predicted_class and confidence > CONFIDENCE_THRESHOLD:
                        current_prediction = predicted_class
                        current_confidence = confidence
            
            # Draw landmarks
            draw_landmarks(frame, results)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                start_time = time.time()
            
            # Draw UI
            # Background for text
            cv2.rectangle(frame, (10, 10), (600, 150), (0, 0, 0), -1)
            
            # Prediction
            color = (0, 255, 0) if current_confidence > CONFIDENCE_THRESHOLD else (0, 165, 255)
            cv2.putText(frame, f"Gesture: {current_prediction}", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            
            # Confidence
            cv2.putText(frame, f"Confidence: {current_confidence:.2%}", (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Buffer status
            buffer_status = f"Buffer: {len(landmark_buffer)}/{BUFFER_SIZE}"
            cv2.putText(frame, buffer_status, (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Instructions
            cv2.putText(frame, "Press 'q' to quit | 'r' to reset", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('GSL Live Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n👋 Quitting...")
                break
            elif key == ord('r'):
                landmark_buffer.clear()
                current_prediction = "Waiting..."
                current_confidence = 0.0
                print("🔄 Buffer reset")
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()
        print("✅ Cleanup complete")


if __name__ == "__main__":
    main()
