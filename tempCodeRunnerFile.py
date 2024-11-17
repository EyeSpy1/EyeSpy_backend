import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from playsound import playsound
import csv
import datetime

# Load the dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define indices for eye landmarks
LEFT_EYE_POINTS = list(range(36, 42))
RIGHT_EYE_POINTS = list(range(42, 48))

def eye_aspect_ratio(eye):
    # Calculate the distances between vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Calculate the distance between horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])

    # Calculate EAR
    ear = (A + B) / (2.0 * C)
    return ear

def log_drowsiness():
    # Log drowsiness event with a timestamp
    with open("drowsiness_log.csv", "a", newline="") as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow([datetime.datetime.now(), "Drowsiness detected"])

def main():
    # Open video capture
    cap = cv2.VideoCapture(0)
    EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold
    EAR_CONSEC_FRAMES = 20  # Number of consecutive frames to detect drowsiness
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            shape = predictor(gray, face)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            left_eye = shape[LEFT_EYE_POINTS]
            right_eye = shape[RIGHT_EYE_POINTS]

            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Check if EAR is below threshold
            if ear < EAR_THRESHOLD:
                frame_counter += 1

                # Trigger alert if eyes are closed for consecutive frames
                if frame_counter >= EAR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    playsound("Drowsiness alert.mp3")
                    log_drowsiness()
                    frame_counter = 0
            else:
                frame_counter = 0

        # Display the frame
        cv2.imshow("Drowsiness Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
