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
from pydub.utils import mediainfo
from pathlib import Path
from pydub import AudioSegment
import hashlib
import pandas as pd
# Constants
st.set_page_config(page_title="Drowsiness Detection", layout="wide")
THRESHOLD = 0.19
FRAME_CHECK = 16
BASE_SAVE_PATH = r"C:\Users\Anushree Jain\Drowsiness"
USER_FOLDER_PATH = os.path.join(BASE_SAVE_PATH, 'user')
ALERT_COOLDOWN = 2



# Initialize session state variables
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
if 'names' not in st.session_state:
    st.session_state.names = []
if 'camera' not in st.session_state:
    st.session_state['camera'] = None
if 'run_detection' not in st.session_state:
    st.session_state.run_detection = False
if 'alert_active' not in st.session_state:
    st.session_state['alert_active'] = False
if 'last_alert_time' not in st.session_state:
    st.session_state.last_alert_time = 0
if 'alert_queue' not in st.session_state:
    st.session_state['alert_queue'] = queue.Queue()
if 'flag' not in st.session_state:
    st.session_state.flag = 0
if 'alert_type' not in st.session_state:
    st.session_state.alert_type = 'tts'
if 'custom_audio_path' not in st.session_state:
    st.session_state.custom_audio_path = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
CREDENTIALS_FILE = os.path.join(BASE_SAVE_PATH, 'user_credentials', 'users.csv')

# Ensure user credentials directory exists
os.makedirs(os.path.dirname(CREDENTIALS_FILE), exist_ok=True)

# Recreate the CSV file with the new structure
def reset_credentials_file():
    try:
        with open(CREDENTIALS_FILE, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['username', 'password_hash', 'email', 'salt'])
        print("Credentials file reset successfully.")
    except Exception as e:
        print(f"Error resetting credentials file: {e}")

def hash_password(password, salt=None):
    """
    Hash the password with a salt
    """
    if salt is None:
        salt = os.urandom(32)  # Generate a random salt
    
    # Hash the password with the salt
    password_hash = hashlib.pbkdf2_hmac(
        'sha256',  # The hash digest algorithm for HMAC
        password.encode('utf-8'),  # Convert the password to bytes
        salt,  # Provide the salt
        100000  # It is recommended to use at least 100,000 iterations of SHA-256 
    )
    
    return salt, password_hash

def validate_signup(username, password, email):
    """
    Validate signup credentials
    Returns: (bool, str) - (is_valid, error_message)
    """
    # Check username
    if not username or len(username) < 3:
        return False, "Username must be at least 3 characters long"
    
    # Check password
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters long"
    
    # Check email (basic validation)
    if not email or '@' not in email or '.' not in email:
        return False, "Invalid email format"
    
    # Check if username already exists
    try:
        users_df = pd.read_csv(CREDENTIALS_FILE)
        if username in users_df['username'].values:
            return False, "Username already exists"
        
        if email in users_df['email'].values:
            return False, "Email already registered"
    except Exception as e:
        return False, f"Error checking credentials: {str(e)}"
    
    return True, "Validation successful"

def signup_user(username, password, email):
    """
    Register a new user with secure password hashing
    Returns: (bool, str) - (is_successful, message)
    """
    # Validate credentials
    is_valid, message = validate_signup(username, password, email)
    if not is_valid:
        return False, message
    
    try:
        # Generate salt and hash the password
        salt, password_hash = hash_password(password)
        
        # Append new user to CSV
        with open(CREDENTIALS_FILE, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                username, 
                password_hash.hex(),  # Convert to hex string for storage
                email, 
                salt.hex()  # Convert salt to hex string for storage
            ])
        
        return True, "Signup successful"
    except Exception as e:
        return False, f"Signup failed: {str(e)}"

def login_user(username, password):
    """
    Authenticate user with secure password checking
    Returns: (bool, str) - (is_authenticated, message)
    """
    try:
        # Read users from CSV
        users_df = pd.read_csv(CREDENTIALS_FILE)
        
        # Find user
        user_row = users_df[users_df['username'] == username]
        
        if user_row.empty:
            return False, "Invalid username or password"
        
        # Retrieve stored salt and password hash
        stored_salt = bytes.fromhex(user_row['salt'].values[0])
        stored_password_hash = bytes.fromhex(user_row['password_hash'].values[0])
        
        # Hash the provided password with the stored salt
        _, input_password_hash = hash_password(password, stored_salt)
        
        # Compare hashes
        if input_password_hash == stored_password_hash:
            return True, "Login successful"
        
        return False, "Invalid username or password"
    except Exception as e:
        # If there's an error with the existing file, reset it
        reset_credentials_file()
        return False, f"Login error: {str(e)}. Please try signing up again."
@st.cache_resource
def load_face_detector():
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        return detector, predictor
    except Exception as e:
        st.error(f"Error loading face detector: {str(e)}")
        return None, None
# Initialize pygame


# Initialize detector and facial landmarks
detector, predictor = load_face_detector()
if detector and predictor:
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
else:
    st.error("Failed to initialize face detector. Please check if the shape predictor file exists.")
    st.stop()
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
try:
    pygame.init()
    pygame.mixer.init()
except pygame.error as e:
    st.error(f"Pygame mixer initialization error: {e}")

def play_alert_sound():
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        
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

def validate_audio_file(uploaded_file):
    """
    Validate the uploaded audio file by checking its format and duration.
    Returns: (bool, str) - (is_valid, error_message)
    """
    try:
        # Create a temporary directory to store the file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as temp_file:
            # Save uploaded file to temporary location
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            # Load the audio file using pydub
            try:
                if file_extension == 'mp3':
                    audio = AudioSegment.from_mp3(temp_file_path)
                elif file_extension == 'wav':
                    audio = AudioSegment.from_wav(temp_file_path)
                elif file_extension == 'mp4':
                    audio = AudioSegment.from_file(temp_file_path, format='mp4')
                else:
                    return False, "Unsupported file format. Please use MP3, WAV, or MP4."
                
                # Check duration (pydub duration is in milliseconds)
                duration_seconds = len(audio) / 1000.0
                
                if duration_seconds <= 0:
                    return False, "Invalid audio file: Duration is 0 seconds"
                elif duration_seconds > 5:
                    return False, f"Audio file too long: {duration_seconds:.1f} seconds. Maximum allowed is 5 seconds"
                
                # Check if file is empty
                if os.path.getsize(temp_file_path) == 0:
                    return False, "Invalid audio file: File is empty"
                
                return True, "Audio file is valid"
                
            except Exception as e:
                return False, f"Error processing audio file: {str(e)}"
            
    except Exception as e:
        return False, f"Error validating audio file: {str(e)}"
    finally:
        # Clean up temporary file
        try:
            if 'temp_file_path' in locals():
                os.unlink(temp_file_path)
        except Exception:
            pass

def cleanup():
    if 'camera' in st.session_state and st.session_state['camera'] is not None:
        st.session_state['camera'].release()
        st.session_state['camera'] = None
    if 'alert_queue' in st.session_state:
        while not st.session_state['alert_queue'].empty():
            # Process or clear the alert queue
            st.session_state['alert_queue'].get()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    st.session_state.run_detection = False
    st.session_state.flag = 0
    st.session_state.alert_active = False
    if 'alert_queue' in st.session_state:
        while not st.session_state['alert_queue'].empty():
            st.session_state['alert_queue'].get()  # Clear queue safely

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


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
    if (not st.session_state['alert_active'] and 
        (current_time - st.session_state.last_alert_time) >= ALERT_COOLDOWN):
        
        st.session_state['alert_active'] = True
        st.session_state.last_alert_time = current_time
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_event(timestamp, user_name)
        
        st.session_state['alert_queue'].put(True)
        play_alert_sound()


def initialize_camera():
    """
    Initialize webcam with deployment-specific configurations
    """
    try:
        # Create placeholder for camera status
        status_placeholder = st.empty()
        status_placeholder.info("⌛ Initializing camera...")
        
        # Initialize camera with specific backend
        camera = cv2.VideoCapture(0, cv2.CAP_ANY)  # Try any available backend
        
        if not camera.isOpened():
            # Try different backend
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # DirectShow (Windows)
            if not camera.isOpened():
                camera = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Video4Linux (Linux)
        
        if not camera.isOpened():
            status_placeholder.error("📛 Camera access failed. Please:")
            st.markdown("""
            1. Refresh the page and allow camera access when prompted
            2. Check if camera is blocked in browser settings
            3. Ensure no other apps are using the camera
            """)
            return None
            
        # Configure camera for web deployment
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        
        # Test camera stream
        ret, frame = camera.read()
        if not ret or frame is None:
            status_placeholder.error("📛 Camera stream failed")
            camera.release()
            return None
        
        status_placeholder.success("✅ Camera ready!")
        return camera
        
    except Exception as e:
        st.error(f"Camera error: {str(e)}")
        if "permission" in str(e).lower():
            st.info("💡 Please allow camera access in your browser")
        return None

def ensure_camera_cleanup():
    """
    Ensure camera is properly released
    """
    if 'camera' in st.session_state and st.session_state['camera'] is not None:
        try:
            st.session_state['camera'].release()
        except:
            pass
        finally:
            st.session_state['camera'] = None
            cv2.destroyAllWindows()
def save_audio_file(user_name, uploaded_file):
    """
    Save the audio file in the user's folder with proper validation and error handling.
    Args:
        user_name (str): The name of the user
        uploaded_file: StreamlitUploadedFile object
    Returns:
        tuple: (str or None, str) - (saved_file_path, error_message)
    """
    try:
        # Validate inputs
        if not user_name or user_name.strip() == "":
            return None, "Please login before uploading audio"
        if not uploaded_file:
            return None, "No file was uploaded"

        # Sanitize user name and create folder path
        safe_name = "".join(x for x in user_name if x.isalnum() or x in (' ', '-', '_')).strip()
        if not safe_name:
            return None, "Invalid user name format"

        # Create user's audio folder path
        user_folder_path = os.path.join(BASE_SAVE_PATH, 'User', safe_name, 'audio')
        os.makedirs(user_folder_path, exist_ok=True)

        # Get file extension
        file_extension = Path(uploaded_file.name).suffix.lower()
        if file_extension not in ['.wav', '.mp3', '.mp4']:
            return None, f"Unsupported file format:{uploaded_file} {file_extension}"

        # Generate unique filename
        new_filename = f"{uploaded_file.name}"
        final_path = os.path.join(user_folder_path, new_filename)

        # Save the file
        with open(final_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Verify file was saved successfully
        if not os.path.exists(final_path):
            return None, "File failed to save"
        if os.path.getsize(final_path) == 0:
            os.remove(final_path)
            return None, "Saved file is empty"

        return final_path, "File saved successfully"

    
    except Exception as e:
        return None, f"Error saving file: {str(e)}"

def main():
    
    # Ensure the credentials file is in the correct format
    try:
        pd.read_csv(CREDENTIALS_FILE)
    except Exception:
        reset_credentials_file()

    st.title("Drowsiness Detection System")
    
    # Authentication method selection
    if 'logged_in_user' not in st.session_state:
        # Authentication method selection (only when not logged in)
        auth_method = st.radio("Select Authentication Method", 
                                ["Login", "Sign Up"], 
                                key="auth_method")
        
        if auth_method == "Login":
            # Login Section
            st.header("Login")
            user_name = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", key="login_button"):
                authenticated, message = login_user(user_name, login_password)
                if authenticated:
                    # Use direct session state modification
                    st.session_state['logged_in_user'] = user_name
                    st.session_state['user_name'] = user_name
                    st.success(message)
                    st.experimental_rerun()
                else:
                    st.error(message)
        
        else:
            # Signup Section
            st.header("Create New Account")
            user_name = st.text_input("Choose Username", key="signup_username")
            new_password = st.text_input("Choose Password", type="password", key="signup_password")
            new_email = st.text_input("Email Address", key="signup_email")
            
            if st.button("Sign Up", key="signup_button"):
                success, message = signup_user(user_name, new_password, new_email)
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    # Detection and logged-in content
    if 'logged_in_user' in st.session_state:
        # Welcome message
        st.write(f"Welcome, {st.session_state.logged_in_user}")
        
        # Logout button
        if st.button("Logout"):
            # Clear the logged-in user state
            del st.session_state['logged_in_user']
            # Optional: clear other session states if needed
            if 'user_name' in st.session_state:
                del st.session_state['user_name']
            
            # Rerun to show login page
            st.experimental_rerun()
        
        # Add your detection logic here (similar to your previous implementation)
    # Add user name input field
    # Display existing names
    

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
                # First validate the file
                is_valid, validation_message = validate_audio_file(uploaded_file)
                
                if is_valid:
                    # Save the file if validation passed
                    saved_path, save_message = save_audio_file(st.session_state.user_name, uploaded_file)
                    
                    if saved_path:
                        st.session_state.custom_audio_path = saved_path
                        st.sidebar.success(f"Using: {os.path.basename(saved_path)}")
                    else:
                        st.sidebar.error(save_message)
                        st.session_state.custom_audio_path = None
                else:
                    st.sidebar.error(validation_message)
    
                    st.session_state.custom_audio_path = None
    if 'logged_in_user' in st.session_state:
        current_log_file = get_log_file_path(st.session_state.user_name)
        st.info(f"Logging to: {os.path.basename(current_log_file)}")
        # Add checkbox to start detection
       

    # Create placeholders for video feed and alerts
    frame_placeholder = st.empty()
    status_placeholder = st.empty()
    alert_placeholder = st.empty()
    start_detection = st.checkbox("Start Detection", key="detection_checkbox")
    if 'run_detection' not in st.session_state:
        st.session_state['run_detection'] = False

    if start_detection and not st.session_state.run_detection:
        if not st.session_state.user_name:
            st.warning("Please login before starting detection.")
            return
        
        st.session_state['camera'] = initialize_camera()
        if st.session_state['camera'] is not None:
            st.session_state.run_detection = True
            st.session_state.flag = 0

    if not start_detection and st.session_state.run_detection:
        cleanup()
        frame_placeholder.empty()
        alert_placeholder.empty()
        return

    while st.session_state.run_detection and st.session_state['camera'] is not None:
        try:
            ret, frame = st.session_state['camera'].read()
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
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 20), 2)
                st.session_state.flag = 0

            frame_placeholder.image(frame, channels="BGR")

            # Fixed: Pass user_name to handle_drowsiness_alert
            if drowsiness_detected and eyes_visible:
                handle_drowsiness_alert(st.session_state.user_name)

            try:
                alert = st.session_state['alert_queue'].get_nowait()
                if alert:
                    alert_placeholder.markdown("""
                    <div style='background-color: #FF4B4B; padding: 20px; border-radius: 10px; text-align: center;'>
                        <h2 style='color: white; margin: 0;'>DROWSINESS ALERT!</h2>
                        <h3 style='color: white; margin: 10px 0;'>Please Wake Up!</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(1)
                    alert_placeholder.empty()
                    st.session_state['alert_active'] = False
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