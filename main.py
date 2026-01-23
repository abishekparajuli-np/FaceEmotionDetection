import os
import cv2
import numpy as np
import tensorflow as tf

# Reduce TensorFlow logs (optional)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ✅ Load your trained model (Windows path safe)
model = tf.keras.models.load_model(r"model\emotion_model (1).keras", compile=False)

# ✅ Emotion labels (MUST match your training order)
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ✅ Load Haar cascade using OpenCV built-in path (NO XML file needed)
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    raise IOError(f"Failed to load Haar cascade from: {face_cascade_path}")

# ✅ Start webcam (CAP_DSHOW fixes Windows black screen)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise IOError("Cannot open webcam. Try changing camera index to 1.")

# Optional: set resolution (faster + stable)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("✅ Webcam started. Press Q or ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # Predict emotion for each face
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]

        # Resize to model input size (48x48)
        face_roi = cv2.resize(face_roi, (48, 48))

        # Normalize and reshape for model
        face_roi = face_roi.astype("float32") / 255.0
        face_roi = np.expand_dims(face_roi, axis=-1)  
        face_roi = np.expand_dims(face_roi, axis=0)   

        # Prediction
        preds = model.predict(face_roi, verbose=0)
        idx = int(np.argmax(preds))
        emotion = emotions[idx]
        confidence = float(preds[0][idx]) * 100

        label = f"{emotion} ({confidence:.1f}%)"

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Emotion Detection (Windows)", frame)

    key = cv2.waitKey(1)
    if key == ord("q") or key == 27:  # q or ESC
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Closed successfully.")
