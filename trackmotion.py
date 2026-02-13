import cv2
import mediapipe as mp
import json
import time

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

motion_data = []

print("Recording... Press 'q' to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    frame_data = {
        "timestamp": time.time(),
        "joints": {}
    }

    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            frame_data["joints"][idx] = {
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z
            }

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    motion_data.append(frame_data)

    cv2.imshow("Dance Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save motion data
with open("dance_motion.json", "w") as f:
    json.dump(motion_data, f)

print("Motion saved to dance_motion.json")