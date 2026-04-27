import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
from collections import Counter

# ===============================
# 1. LOAD MODEL
# ===============================
print("Loading model...")
model = load_model("UPDATED_baseline_phase-two.keras")

classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

emotion_colors = {
    'angry': (0, 0, 255),
    'disgust': (0, 100, 0),
    'fear': (0, 0, 0),
    'happy': (0, 255, 255),
    'sad': (255, 0, 0),
    'surprise': (255, 0, 255),
    'neutral': (255, 255, 255)
}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

# ===============================
# 2. OPTIMIZATION VARIABLES
# ===============================
frame_skip = 3
frame_count = 0
last_detections = []

# FPS tracking
prev_time = time.time()

# Emotion tracking
emotion_history = []
confidence_history = []

print("Press ESC to exit...")

# ===============================
# 3. MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if frame_count % frame_skip == 0:
        faces = face_cascade.detectMultiScale(
            gray, 1.3, 5, minSize=(30, 30)
        )

        last_detections = []

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (48, 48))
            face_normalized = face_resized / 255.0

            face_input = np.expand_dims(face_normalized, axis=(0, -1))

            pred = model(face_input, training=False).numpy()[0]

            emotion_idx = np.argmax(pred)
            label = classes[emotion_idx]
            confidence = pred[emotion_idx] * 100

            # Save history
            emotion_history.append(label)
            confidence_history.append(confidence)

            last_detections.append((x, y, w, h, label, confidence))

    frame_count += 1

    # ===============================
    # DRAW UI
    # ===============================
    for (x, y, w, h, label, confidence) in last_detections:
        color = emotion_colors.get(label, (0, 255, 0))

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        text = f"{label.upper()} {confidence:.0f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX

        (text_w, text_h), _ = cv2.getTextSize(text, font, 0.6, 2)

        cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y), color, -1)

        text_color = (0, 0, 0) if label in ['happy', 'neutral'] else (255, 255, 255)

        cv2.putText(frame, text, (x, y - 5), font, 0.6, text_color, 2)

    # ===============================
    # FPS DISPLAY
    # ===============================
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Emotion AI - Optimized", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ===============================
# 4. CLEANUP
# ===============================
cap.release()
cv2.destroyAllWindows()

# ===============================
# 5. FINAL SUMMARY WINDOW
# ===============================
print("Generating summary...")

if len(emotion_history) > 0:

    most_common_emotion = Counter(emotion_history).most_common(1)[0][0]
    avg_confidence = sum(confidence_history) / len(confidence_history)

    # Create summary image
    summary_img = np.zeros((400, 600, 3), dtype=np.uint8)

    cv2.putText(summary_img, "SESSION SUMMARY", (120, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.putText(summary_img, f"Total Frames Analyzed: {len(emotion_history)}",
                (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(summary_img, f"Dominant Emotion: {most_common_emotion.upper()}",
                (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(summary_img, f"Average Confidence: {avg_confidence:.2f}%",
                (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Emotion distribution
    y_offset = 300
    counts = Counter(emotion_history)

    for emo, count in counts.items():
        cv2.putText(summary_img, f"{emo}: {count}",
                    (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1)
        y_offset += 30

    # Show summary window
    cv2.imshow("Session Summary", summary_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("No emotions detected.")
    