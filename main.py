#Github:
#*
# git add .
# git commit -m "Describe your change here"
# git push
# *#

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Define arm landmarks
arm_indices = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.RIGHT_WRIST,
]

# Define arm connections
arm_connections = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        full_landmarks = results.pose_landmarks.landmark

        # Create filtered landmark list
        filtered_landmarks = landmark_pb2.NormalizedLandmarkList()
        index_map = {}  # maps original index to new index

        for i, idx in enumerate(arm_indices):
            filtered_landmarks.landmark.append(full_landmarks[idx])
            index_map[idx] = i

        # Remap connections to new indices
        filtered_connections = [
            (index_map[start], index_map[end])
            for start, end in arm_connections
        ]

        mp_drawing.draw_landmarks(
            frame,
            filtered_landmarks,
            filtered_connections,
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2),
        )

    cv2.imshow('Just Arms (Clean)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
