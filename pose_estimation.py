import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=True)

def detect_pose(image):

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    landmarks_vector = []

    if results.pose_landmarks:

        # draw skeleton
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # extract landmark coordinates
        for lm in results.pose_landmarks.landmark:
            landmarks_vector.append(lm.x)
            landmarks_vector.append(lm.y)

    return image, np.array(landmarks_vector)