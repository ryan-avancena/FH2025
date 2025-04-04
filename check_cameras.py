import cv2

for i in range(5):  # Test up to 5 possible camera indexes
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        cap.release()