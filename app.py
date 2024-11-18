import streamlit as st
import os
import uuid 
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
import pygame
import wave
import contextlib
import tempfile
import shutil

# Initialize session state variables
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
if 'names' not in st.session_state:
    st.session_state.names = []
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
if 'alert_type' not in st.session_state:
    st.session_state.alert_type = 'tts'
if 'custom_audio_path' not in st.session_state:
    st.session_state.custom_audio_path = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# Constants
THRESHOLD = 0.19
FRAME_CHECK = 20
BASE_SAVE_PATH = r"C:\Users\Anushree Jain\Drowsiness"
USER_FOLDER_PATH = os.path.join(BASE_SAVE_PATH, 'user')
ALERT_COOLDOWN = 3

# Create base save directory
try:
    os.makedirs(BASE_SAVE_PATH, exist_ok=True)
except PermissionError:
    BASE_SAVE_PATH = os.path.join(os.path.expanduser("~"), "temp_drowsiness_logs")
    try:
        os.makedirs(BASE_SAVE_PATH, exist_ok=True)
    except:
        st.error("Unable to create log directory. Please check permissions.")

def get_log_file_path(user_name):
    if user_name:
        safe_name = "".join(x for x in user_name if x.isalnum() or x in (' ', '-', '_')).strip()
        if safe_name:
            user_folder_path = os.path.join(BASE_SAVE_PATH, 'User', safe_name)
            os.makedirs(user_folder_path, exist_ok=True)
            return os.path.join(user_folder_path, f"{safe_name}_drowsiness_log.csv")
    return os.path.join(BASE_SAVE_PATH, "drowsiness_log.csv")

def get_tts_engine():
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        return engine
    except Exception as e:
        st.error(f"Error initializing TTS engine: {str(e)}")
        return None

pygame.init()
pygame.mixer.init()

def play_alert_sound():
    try:
        if st.session_state.alert_type == 'tts':
            engine = get_tts_engine()
            if engine:
                engine.say("Drowsiness alert! Please stay awake!")
                engine.runAndWait()
                engine.stop()
                del engine
        elif st.session_state.alert_type == 'custom' and st.session_state.custom_audio_path:
            pygame.mixer.music.load(st.session_state.custom_audio_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
    except Exception as e:
        st.error(f"Error playing sound: {str(e)}")

def validate_audio_file(file):
    try:
        with contextlib.closing(wave.open(file, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return duration <= 5
    except Exception:
        return False

def cleanup():
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    st.session_state.run_detection = False
    st.session_state.flag = 0
    st.session_state.alert_active = False
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

@st.cache_resource
def load_face_detector():
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        return detector, predictor
    except Exception as e:
        st.error(f"Error loading face detector: {str(e)}")
        return None, None

def log_event(timestamp, user_name):
    try:
        log_file_path = get_log_file_path(user_name)
        
        dt = datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")
        date = dt.strftime("%Y-%m-%d")
        time = dt.strftime("%H:%M:%S")
        
        if not os.path.exists(log_file_path):
            with open(log_file_path, "w", newline="") as log_file:
                writer = csv.writer(log_file)
                writer.writerow(["Date", "Time", "Event"])
        
        existing_dates = set()
        with open(log_file_path, "r") as log_file:
            reader = csv.reader(log_file)
            next(reader)
            for row in reader:
                if row:
                    existing_dates.add(row[0])
        
        with open(log_file_path, "a", newline="") as log_file:
            writer = csv.writer(log_file)
            if date not in existing_dates:
                writer.writerow([date, "", ""]) 
                writer.writerow([date, time, "New Date Entry"])
            writer.writerow([date, time, "Drowsiness Detected"])
    
    except Exception as e:
        st.error(f"Error logging event: {str(e)}")

def handle_drowsiness_alert(user_name):
    current_time = time.time()
    if (not st.session_state.alert_active and 
        (current_time - st.session_state.last_alert_time) >= ALERT_COOLDOWN):
        
        st.session_state.alert_active = True
        st.session_state.last_alert_time = current_time
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_event(timestamp, user_name)
        
        st.session_state.alert_queue.put(True)
        play_alert_sound()

def initialize_camera():
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            st.error("Error: Unable to access webcam. Please check permissions and connections.")
            return None
        return camera
    except Exception as e:
        st.error(f"Error initializing camera: {str(e)}")
        return None

def save_audio_file(user_name, uploaded_file):
    """
    Save the audio file inside the user's folder, checking for duplicates by content.
    Only saves if the file content is different from existing files.
    """
    try:
        if not user_name or not uploaded_file:
            return None
            
        # Create user folder path
        safe_name = "".join(x for x in user_name if x.isalnum() or x in (' ', '-', '_')).strip()
        user_folder_path = os.path.join(BASE_SAVE_PATH, 'User', safe_name)
        os.makedirs(user_folder_path, exist_ok=True)
        
        # Get original filename and content
        original_filename = uploaded_file.name
        file_content = uploaded_file.getvalue()
        
        # Check if file with same content already exists
        for existing_file in os.listdir(user_folder_path):
            existing_path = os.path.join(user_folder_path, existing_file)
            if os.path.isfile(existing_path):
                try:
                    with open(existing_path, 'rb') as f:
                        existing_content = f.read()
                        if existing_content == file_content:
                            # File with same content already exists, return its path
                            return existing_path
                except Exception:
                    continue
        
        # If no duplicate found, save the new file
        filename, extension = os.path.splitext(original_filename)
        counter = 0
        final_path = os.path.join(user_folder_path, original_filename)
        
        while os.path.exists(final_path):
            counter += 1
            new_filename = f"{filename}_{counter}{extension}"
            final_path = os.path.join(user_folder_path, new_filename)
        
        with open(final_path, "wb") as f:
            f.write(file_content)
            
        return final_path
        
    except Exception as e:
        st.error(f"Error saving audio file: {str(e)}")
        return None

# Add to session state initializations at the start of the script
if 'last_uploaded_content' not in st.session_state:
    st.session_state.last_uploaded_content = None

# Initialize detector and facial landmarks
detector, predictor = load_face_detector()
if detector and predictor:
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
else:
    st.error("Failed to initialize face detector. Please check if the shape predictor file exists.")
    st.stop()

def main():
    st.title("Drowsiness Detection System")
    st.write("This application detects drowsiness using your webcam.")

    # Add user name input field
    user_name = st.text_input("Enter your name:", key="user_name_input")
    if user_name:
        st.session_state.user_name = user_name
        if user_name not in st.session_state.names:
            st.session_state.names.append(user_name)

    # Display existing names
    if st.session_state.names:
        selected_name = st.selectbox("Or select an existing name:", 
                                   [""] + st.session_state.names,
                                   key="name_selector")
        if selected_name:
            st.session_state.user_name = selected_name

    # Audio alert settings
    st.sidebar.header("Alert Settings")
    alert_type = st.sidebar.radio(
        "Choose alert type:",
        ('Text-to-Speech', 'Custom Audio'),
        key='alert_type_radio'
    )
    
    st.session_state.alert_type = 'tts' if alert_type == 'Text-to-Speech' else 'custom'
    
    # Initialize uploaded_file in session state
    if st.session_state.alert_type == 'custom':
        uploaded_file = st.sidebar.file_uploader(
            "Upload alert sound (WAV format, max 5 seconds)",
            type=['wav', 'mp3', 'mp4'],
            key='audio_uploader'
        )
    
        if uploaded_file is not None:
            # Check if this is a new file content
            current_content = uploaded_file.getvalue()
            if st.session_state.last_uploaded_content != current_content:
                try:
                    saved_path = save_audio_file(st.session_state.user_name, uploaded_file)
                    
                    if saved_path and validate_audio_file(saved_path):
                        st.session_state.custom_audio_path = saved_path
                        st.session_state.last_uploaded_content = current_content
                        st.sidebar.success(f"Using audio file: {os.path.basename(saved_path)}")
                    else:
                        st.sidebar.error("Invalid audio file. Please ensure it's a valid audio file under 5 seconds.")
                        st.session_state.custom_audio_path = None
                        if saved_path and os.path.exists(saved_path):
                            os.remove(saved_path)
                except Exception as e:
                    st.sidebar.error(f"Error processing audio file: {str(e)}")

    if st.session_state.user_name:
        current_log_file = get_log_file_path(st.session_state.user_name)
        st.info(f"Logging to: {os.path.basename(current_log_file)}")

    # Create placeholders for video feed and alerts
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    alert_placeholder = st.empty()

    # Add checkbox to start detection
    start_detection = st.checkbox("Start Detection", key="detection_checkbox")

    if start_detection and not st.session_state.run_detection:
        if not st.session_state.user_name:
            st.warning("Please enter your name before starting detection.")
            return
        
        st.session_state.camera = initialize_camera()
        if st.session_state.camera is not None:
            st.session_state.run_detection = True
            st.session_state.flag = 0

    if not start_detection and st.session_state.run_detection:
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
            eyes_visible = False

            if len(subjects) == 0:
                cv2.putText(frame, "No face detected", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 69, 139), 2)
                st.session_state.flag = 0
            else:
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

                    eyes_visible = True
                    if ear < THRESHOLD:
                        st.session_state.flag += 1
                        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        if st.session_state.flag >= FRAME_CHECK:
                            drowsiness_detected = True
                            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        st.session_state.flag = 0
                        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if not eyes_visible:
                cv2.putText(frame, "No face detected", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                st.session_state.flag = 0

            frame_placeholder.image(frame, channels="BGR")

            # Fixed: Pass user_name to handle_drowsiness_alert
            if drowsiness_detected and eyes_visible:
                handle_drowsiness_alert(st.session_state.user_name)

            try:
                alert = st.session_state.alert_queue.get_nowait()
                if alert:
                    alert_placeholder.markdown("""
                    <div style='background-color: #FF4B4B; padding: 20px; border-radius: 10px; text-align: center;'>
                        <h2 style='color: white; margin: 0;'>DROWSINESS ALERT!</h2>
                        <h3 style='color: white; margin: 10px 0;'>Please Wake Up!</h3>
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