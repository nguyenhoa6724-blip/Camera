import cv2
import mediapipe as mp
import numpy as np
import math

# -----------------------------
# EAR & MAR calculation
# -----------------------------
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)



# -----------------------------
# Mediapipe init
# -----------------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark index (MediaPipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 81, 13, 311, 308, 402]

# Threshold
EAR_TH = 0.21
MAR_TH = 0.70

# -----------------------------
# Camera
# -----------------------------
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face in result.multi_face_landmarks:
            landmarks = face.landmark

            left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]
            mouth = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in MOUTH]

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2
            mar = mouth_aspect_ratio(mouth)

            # Status
            eye_status = "CLOSED" if ear < EAR_TH else "OPEN"
            yawn_status = "YES" if mar > MAR_TH else "NO"

            # Draw
            cv2.putText(frame, f"EAR: {ear:.2f} Eyes: {eye_status}",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if eye_status=="CLOSED" else (0,255,0), 2)

            cv2.putText(frame, f"MAR: {mar:.2f} Yawn: {yawn_status}",
                        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if yawn_status=="YES" else (0,255,0), 2)

            # Draw landmarks
            for p in left_eye + right_eye + mouth:
                cv2.circle(frame, p, 2, (255, 0, 0), -1)

    cv2.imshow("Driver Monitoring Demo", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
