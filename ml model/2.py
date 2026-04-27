import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("UPDATED_baseline_phase-two.keras")

classes = ['angry','disgust','fear','happy','sad','surprise','neutral']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

for _ in range(300):   # limited run

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48))

        face = face / 255.0
        face = face.reshape(1,48,48,1)

        pred = model.predict(face, verbose=0)
        label = classes[np.argmax(pred)]

        # 🔥 draw box
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        # 🔥 show label
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0,255,0), 2)

    cv2.imshow("Emotion AI", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()