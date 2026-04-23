import cv2
import os
from datetime import datetime
import time

# Create folder for saving images
save_folder = "detected_faces"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Load Haar Cascade model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=6,
        minSize=(50, 50)
    )

    # FPS Calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Title
    cv2.putText(
        frame,
        "Face Detection System (OpenCV AI)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # Face Counter
    cv2.putText(
        frame,
        f"Faces Detected: {len(faces)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    # FPS Display
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Detect faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.putText(
            frame,
            "Face Detected",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Save screenshot on key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_folder}/face_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print("Screenshot Saved:", filename)

    if key == ord('c'):  # Clear folder
        for file in os.listdir(save_folder):
            os.remove(os.path.join(save_folder, file))
        print("All images cleared!")

    if key == ord('q'):  # Quit
        break

    cv2.imshow("Face Detection AI System", frame)

cap.release()
cv2.destroyAllWindows()