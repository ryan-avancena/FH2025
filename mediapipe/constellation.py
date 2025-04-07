import mediapipe as mp
import numpy as np
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Drawing settings
contour_color = (255, 255, 255)  # Glowy yellowish for starry effect

# Capture from webcam
cap = cv2.VideoCapture(1)

def draw_constellation(image, landmarks, connections, color=(0, 0, 0), radius=1):
    h, w, _ = image.shape

    # Draw connecting lines
    for connection in connections:
        start_idx, end_idx = connection
        start = landmarks.landmark[start_idx]
        end = landmarks.landmark[end_idx]

        x1, y1 = int(start.x * w), int(start.y * h)
        x2, y2 = int(end.x * w), int(end.y * h)
        cv2.line(image, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)

    # Draw stars (dots)
    for lm in landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (x, y), radius, color, -1, lineType=cv2.LINE_AA)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip and process frame
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        # Black canvas
        image = np.zeros_like(frame)

        # Draw constellation if face detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                draw_constellation(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, color=(180, 180, 255), radius=1)
                draw_constellation(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, color=contour_color, radius=2)

        # Add glow
        glow = cv2.GaussianBlur(image, (0, 0), sigmaX=7, sigmaY=7)
        image = cv2.addWeighted(image, 1.0, glow, 0.8, 0)

        # Show output
        cv2.imshow("Face Constellation", image)

        # Exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
