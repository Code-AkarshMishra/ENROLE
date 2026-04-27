import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


print("Loading model...")
model = load_model("UPDATED_baseline_phase-two.keras")
classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# 2. Map emotions to specific BGR colors for the UI
emotion_colors = {
    'angry': (0, 0, 255),      # Red
    'disgust': (0, 100, 0),    # Dark Green
    'fear': (0, 0, 0),         # Black
    'happy': (0, 255, 255),    # Yellow
    'sad': (255, 0, 0),        # Blue
    'surprise': (255, 0, 255), # Magenta
    'neutral': (255, 255, 255) # White
}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

# Optimization variables
frame_skip = 3       # Run prediction every 3rd frame to increase FPS
frame_count = 0
last_detections = [] # Store results to draw during skipped frames

print("Starting video stream. Press 'ESC' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for a natural mirror effect
    frame = cv2.flip(frame, 1) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Only run the heavy detection/prediction every few frames
    if frame_count % frame_skip == 0:
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.3, 
            minNeighbors=5, 
            minSize=(30, 30) # Ignore tiny false positives
        )
        
        last_detections = [] # Clear old detections

        for (x, y, w, h) in faces:
            # Extract and preprocess the face
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (48, 48))
            face_normalized = face_resized / 255.0
            
            # Reshape using expand_dims for cleaner syntax: (1, 48, 48, 1)
            face_input = np.expand_dims(face_normalized, axis=(0, -1))

            # 🔥 OPTIMIZATION: model() is much faster than model.predict() for single inputs
            pred = model(face_input, training=False).numpy()[0]
            
            emotion_idx = np.argmax(pred)
            label = classes[emotion_idx]
            confidence = pred[emotion_idx] * 100
            
            last_detections.append((x, y, w, h, label, confidence))

    frame_count += 1

    # Draw the UI based on the most recent detections
    for (x, y, w, h, label, confidence) in last_detections:
        color = emotion_colors.get(label, (0, 255, 0)) # Default to green if missing

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Prepare text and get its size for the background box
        text = f"{label.upper()} {confidence:.0f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(text, font, 0.6, 2)

        # Draw solid background for text
        cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y), color, -1)
        
        # Determine text color (black for light backgrounds, white for dark)
        text_color = (0, 0, 0) if label in ['happy', 'neutral'] else (255, 255, 255)

        # Draw text
        cv2.putText(frame, text, (x, y - 5), font, 0.6, text_color, 2)

    cv2.imshow("Emotion AI - Optimized", frame)

    # Exit on 'ESC' key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()