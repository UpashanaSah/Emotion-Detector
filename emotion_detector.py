import cv2
import numpy as np
from keras.models import load_model

# Load model
model = load_model("emotion_model.h5")

# Emotion labels (FER2013 order)
emotion_labels = [
    'angry',
    'disgust',
    'fear',
    'happy',
    'neutral',
    'sad',
    'surprise'
]


# Load face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        padding = 10
        roi_gray = gray[
            max(0, y-padding):min(gray.shape[0], y+h+padding),
            max(0, x-padding):min(gray.shape[1], x+w+padding)]

        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray / 255.0
        roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))

        preds = model.predict(roi_gray, verbose=0)
        emotion = emotion_labels[np.argmax(preds)]
        confidence = np.max(preds)
        print(emotion, confidence)


        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"{emotion} ({confidence:.2f})",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
            )


    cv2.imshow("Emotion Detector by Upashana", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


