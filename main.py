import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

NUM_CLASSES = 7

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

ema_state = torch.load("best_rafdb_resnet50.pth", map_location=DEVICE)

missing, unexpected = model.load_state_dict(ema_state, strict=False)

print("Missing keys (expected):", len(missing))
print("Unexpected keys:", len(unexpected))

model = model.to(DEVICE)
model.eval()

emotion_labels = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happiness",
    4: "Sadness",
    5: "Surprise",
    6: "Neutral"
}

# EMOTION → STRESS MAP (0–100)

emotion_stress_map = {
    0: 80,   # Anger
    1: 75,   # Disgust
    2: 90,   # Fear
    3: 10,   # Happiness
    4: 70,   # Sadness
    5: 50,   # Surprise
    6: 35    # Neutral
}
# CAMERA
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("ERROR: Cannot open camera")

print("Press Q to exit")

stress_history = []

# MAIN LOOP
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    frame_stress_values = []

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))

        face_tensor = torch.tensor(face).permute(2, 0, 1)
        face_tensor = face_tensor.unsqueeze(0).float() / 255.0
        face_tensor = face_tensor.to(DEVICE)

        with torch.no_grad():
            outputs = model(face_tensor)
            pred_class = outputs.argmax(dim=1).item()

        emotion = emotion_labels[pred_class]
        stress_value = emotion_stress_map[pred_class]

        frame_stress_values.append(stress_value)

        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Smooth stress over time
    if frame_stress_values:
        stress_history.append(np.mean(frame_stress_values))
        if len(stress_history) > 30:
            stress_history.pop(0)

    final_stress = np.mean(stress_history) if stress_history else 0

    cv2.putText(
        frame,
        f"Stress Level: {final_stress:.1f} / 100",
        (10, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2
    )

    cv2.imshow("Emotion-Based Stress Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break

# CLEANUP
cap.release()
cv2.destroyAllWindows()
