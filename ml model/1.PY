import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("UPDATED_baseline_phase-two.keras")

classes = ['angry','disgust','fear','happy','sad','surprise','neutral']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face = cv2.resize(gray, (48,48))
    face = face / 255.0
    face = face.reshape(1,48,48,1)

    pred = model.predict(face)
    label = classes[np.argmax(pred)]

    cv2.putText(frame, label, (30,50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()