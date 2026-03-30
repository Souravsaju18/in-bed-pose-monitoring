import cv2
import os
import numpy as np
import mediapipe as mp
import joblib
from sklearn.ensemble import RandomForestClassifier

# Mediapipe pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Dataset path
dataset_root = "/Users/souravsaju/Downloads/archive/train/train"

X = []
y = []

print("Scanning dataset...")

for subject in os.listdir(dataset_root):

    subject_path = os.path.join(dataset_root, subject)

    rgb_path = os.path.join(subject_path, "RGB", "uncover")

    if not os.path.exists(rgb_path):
        continue

    images = os.listdir(rgb_path)

    for img_name in images:

        img_path = os.path.join(rgb_path, img_name)

        image = cv2.imread(img_path)

        if image is None:
            continue

        # Convert to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb)

        if not results.pose_landmarks:
            continue

        features = []

        for lm in results.pose_landmarks.landmark[:33]:
            features.append(lm.x)
            features.append(lm.y)

        # --- Label Assignment ---
        # We simulate posture labels based on subject ID
        subject_id = int(subject)

        if subject_id <= 30:
            label = 0  # Supine
        elif subject_id <= 55:
            label = 1  # Left Side
        else:
            label = 2  # Right Side

        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Feature shape:", X.shape)

# Train Random Forest
print("Training model...")

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Save model
joblib.dump(model, "posture_model.pkl")

print("Model saved as posture_model.pkl")