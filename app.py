import streamlit as st
import os
import time
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import csv
from datetime import datetime
import pyttsx3
import numpy as np
from threading import Thread
import queue

# Initialize text-to-speech engine in a thread-safe way
def get_tts_engine():
    """Create a new TTS engine instance each time"""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        return engine
    except Exception as e:
        st.error(f"Error initializing TTS engine: {str(e)}")
        return None

# Initialize session state variables
if 'camera' not in st.session_state:
    st.session_state.camera = None
if 'run_detection' not in st.session_state:
    st.session_state.run_detection = False
if 'alert_active' not in st.session_state:
    st.session_state.alert_active = False
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = 0
if 'alert_queue' not in st.session_state:
    st.session_state.alert_queue = queue.Queue()
if 'flag' not in st.session_state:
    st.session_state.flag = 0

def play_alert_sound():
    """Play alert sound using a new TTS engine instance"""
    try:
        engine = get_tts_engine()
        if engine:
            engine.say("Drowsiness alert! Please stay awake!")
            engine.runAndWait()
            engine.stop()
            del engine
    except Exception as e:
        st.error(f"Error playing sound: {str(e)}")

def cleanup():
    """Cleanup function to release resources"""
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None
    cv2.destroyAllWindows()
    st.session_state.run_detection = False
    st.session_state.flag = 0
    st.session_state.alert_active = False
    # Clear the alert queue
    while not st.session_state.alert_queue.empty():
        try:
            st.session_state.alert_queue.get_nowait()
        except queue.Empty:
            break

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Define constants
THRESHOLD = 0.25
FRAME_CHECK = 20
SAVE_PATH = os.path.join(os.path.expanduser("~"), "drowsiness_logs")
LOG_FILE = os.path.join(SAVE_PATH, "drowsiness_log.csv")
ALERT_COOLDOWN = 3

# Ensure save directory exists
os.makedirs(SAVE_PATH, exist_ok=True)

# Initialize Dlib's face detector and predictor
@st.cache_resource
def load_face_detector():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return detector, predictor

detector, predictor = load_face_detector()
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

def log_event(image, timestamp):
    """Log drowsiness events"""
    image_filename = os.path.join(SAVE_PATH, f"drowsiness_{timestamp}.png")
    cv2.imwrite(image_filename, image)
    
    with open(LOG_FILE, "a", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow([timestamp, "Drowsiness Detected", image_filename])

def handle_drowsiness_alert(frame):
    """Handle the complete drowsiness alert sequence"""
    current_time = time.time()
    if (not st.session_state.alert_active and 
        (current_time - st.session_state.last_alert_time) >= ALERT_COOLDOWN):
        
        st.session_state.alert_active = True
        st.session_state.last_alert_time = current_time
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_event(frame, timestamp)
        
        # Add alert to queue
        st.session_state.alert_queue.put(True)
        
        # Play sound in main thread to avoid Streamlit threading issues
        play_alert_sound()

def initialize_camera():
    """Initialize the camera"""
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            st.error("Error: Unable to access webcam.")
            return None
        return camera
    except Exception as e:
        st.error(f"Error initializing camera: {str(e)}")
        return None

def main():
    st.title("Drowsiness Detection")
    st.write("This application detects drowsiness using your webcam.")

    # Create placeholders
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    alert_placeholder = st.empty()

    # Add checkbox to start detection
    start_detection = st.checkbox("Start Detection", key="detection_checkbox")

    if start_detection and not st.session_state.run_detection:
        # Initialize camera only when starting detection
        st.session_state.camera = initialize_camera()
        if st.session_state.camera is not None:
            st.session_state.run_detection = True
            st.session_state.flag = 0

    if not start_detection and st.session_state.run_detection:
        # Cleanup when stopping detection
        cleanup()
        frame_placeholder.empty()
        alert_placeholder.empty()
        return

    while st.session_state.run_detection and st.session_state.camera is not None:
        try:
            ret, frame = st.session_state.camera.read()
            if not ret:
                st.error("Error: Failed to grab frame from webcam.")
                cleanup()
                break

            frame = imutils.resize(frame, width=720)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            subjects = detector(gray, 0)

            drowsiness_detected = False

            for subject in subjects:
                shape = predictor(gray, subject)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                if ear < THRESHOLD:
                    st.session_state.flag += 1
                    if st.session_state.flag >= FRAME_CHECK:
                        drowsiness_detected = True
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    st.session_state.flag = 0

            frame_placeholder.image(frame, channels="BGR")

            if drowsiness_detected:
                handle_drowsiness_alert(frame)

            # Check for alerts in queue and display them
            try:
                alert = st.session_state.alert_queue.get_nowait()
                if alert:
                    alert_placeholder.markdown("""
                    <div style='background-color: #FF4B4B; padding: 20px; border-radius: 10px; text-align: center;'>
                        <h2 style='color: white; margin: 0;'>ALERT!</h2>
                        <h3 style='color: white; margin: 10px 0;'>Wake Up!</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(2)
                    alert_placeholder.empty()
                    st.session_state.alert_active = False
            except queue.Empty:
                pass

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            cleanup()
            break

if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()