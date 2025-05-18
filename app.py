import os
import io
import cv2
import dlib
import numpy as np
import sqlite3
import hashlib
import streamlit as st
from scipy.spatial import distance as dist
from pydub import AudioSegment
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoHTMLAttributes,
    AudioHTMLAttributes,
)
import av

# --- DB and Model Setup ---
DB_PATH = "users.db"
USER_DATA_DIR = "user_data"
MODEL_PATH = os.path.join("models", "shape_predictor_68_face_landmarks.dat")
os.makedirs("models", exist_ok=True)
os.makedirs(USER_DATA_DIR, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            email TEXT,
            hashed TEXT,
            salt TEXT
        )
    """)
    conn.commit()
    conn.close()
init_db()

def get_user(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT username, email, hashed, salt FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    return row

def add_user(username, email, hashed, salt):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO users (username, email, hashed, salt) VALUES (?, ?, ?, ?)", (username, email, hashed, salt))
    conn.commit()
    conn.close()

def email_exists(email):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT 1 FROM users WHERE email = ?", (email,))
    exists = c.fetchone() is not None
    conn.close()
    return exists

def hash_password(password, salt=None):
    salt = salt or os.urandom(16).hex()
    hashed = hashlib.sha256((password + salt).encode()).hexdigest()
    return hashed, salt

def verify_password(password, salt, hashed):
    return hashlib.sha256((password + salt).encode()).hexdigest() == hashed

# --- Dlib Model ---
try:
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(MODEL_PATH)
    face_detection_ready = True
except Exception as e:
    st.error("Landmark model not found or corrupted. Download from [dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), extract, and put in 'models' folder.")
    face_detection_ready = False

# --- EAR Calculation ---
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C else 0

# --- Streamlit Session State ---
for key, val in [
    ("authenticated", False), ("username", ""), ("counter", 0), ("alarm_on", False),
    ("debug", False), ("ear_values", []), ("alarm_audio", None), ("alarm_array", None),
    ("current_ear", 0.0), ("left_ear", 0.0), ("right_ear", 0.0), ("pygame_init", False),
    ("previous_alarm_state", False)  # Add this to track alarm state changes
]:
    if key not in st.session_state:
        st.session_state[key] = val

# --- Auth UI ---
if not st.session_state.authenticated:
    st.title("👁️ Driver Drowsiness Detector")
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    with tab1:
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user_row = get_user(username)
            if user_row:
                _, _, hashed, salt = user_row
                if verify_password(password, salt, hashed):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password.")
            else:
                st.error("Invalid username or password.")
    with tab2:
        st.subheader("Create Account")
        new_username = st.text_input("Username", key="new_user")
        new_email = st.text_input("Email", key="new_email")
        new_password = st.text_input("Password", type="password", key="new_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        if st.button("Sign Up"):
            if not new_username or not new_email or not new_password:
                st.error("All fields are required.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif get_user(new_username):
                st.error("Username already exists.")
            elif email_exists(new_email):
                st.error("Email already registered.")
            else:
                hashed_pw, salt = hash_password(new_password)
                add_user(new_username, new_email, hashed_pw, salt)
                st.success("Account created! Please login.")

# --- Main App ---
else:
    st.title("👁️ Driver Drowsiness Detector")
    st.markdown(f"**Logged in as:** `{st.session_state.username}`")
    if not face_detection_ready:
        st.error("Face detection is not working. Please check the model file.")
        st.stop()

    user_dir = os.path.join(USER_DATA_DIR, st.session_state.username)
    os.makedirs(user_dir, exist_ok=True)

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")
        EAR_THRESH = st.slider("Eye Aspect Ratio Threshold", 0.15, 0.40, 0.25, 0.01)
        CONSEC_FRAMES = st.slider("Consecutive Frames", 5, 30, 15, 1)
        ALARM_VOLUME = st.slider("Alarm Volume", 0.5, 2.0, 1.5, 0.1)  # Higher max volume (2.0)
        st.session_state.debug = st.checkbox("Debug Mode", value=False)
        st.write("---")
        
        if st.button("Log Out"):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.experimental_rerun()

    # --- Check if beep.wav exists ---
    beep_file = "beep.wav"
    if not os.path.exists(beep_file):
        st.warning("beep.wav file not found. Please add it to the application directory.")
        
    # Import pygame for audio playback (as a backup method)
    try:
        import pygame
    except ImportError:
        st.error("Pygame is required to play beep.wav. Install it using 'pip install pygame'.")
        
    # Create enhanced alarm sound for WebRTC audio callback
    def create_beep_sound(duration_secs=1.0, frequency=1000, sample_rate=44100, volume=1.5):
        t = np.linspace(0, duration_secs, int(sample_rate * duration_secs), False)
        
        # Create a more attention-grabbing sound with multiple frequencies
        beep = np.sin(2 * np.pi * frequency * t) * volume * 32767
        beep += np.sin(2 * np.pi * frequency * 1.5 * t) * volume * 0.7 * 32767  # Add harmonic
        beep += np.sin(2 * np.pi * frequency * 2.0 * t) * volume * 0.4 * 32767  # Add another harmonic
        
        # Create a more urgent pattern with shorter intervals
        beep_pattern = np.zeros(int(sample_rate * duration_secs), dtype=np.int16)
        interval = int(sample_rate * 0.15)  # Shorter interval (0.15s instead of 0.2s)
        for i in range(0, len(beep_pattern), interval*2):
            if i + interval <= len(beep_pattern):
                beep_pattern[i:i+interval] = beep[i:i+interval].astype(np.int16)
                
        # Apply volume scaling based on user setting
        beep_pattern = (beep_pattern * ALARM_VOLUME).astype(np.int16)
        
        return beep_pattern

    # Create alarm sound with higher volume and urgency
    alarm_sound = create_beep_sound(duration_secs=2.0, frequency=880, volume=1.5)
    st.session_state.alarm_array = alarm_sound

    # --- Video Frame Callback ---
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]
        faces = detector(gray, 0)
        ear = 0.0
        left_ear = 0.0
        right_ear = 0.0

        if not faces:
            st.session_state.counter = 0
            st.session_state.alarm_on = False
        for face in faces:
            shape = predictor(gray, face)
            shape_np = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
            for (x, y) in shape_np:
                cv2.circle(img, (x, y), 2, (0, 255, 255), -1)  # Yellow dots
            left_eye = shape_np[42:48]
            right_eye = shape_np[36:42]
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            st.session_state.left_ear = left_ear
            st.session_state.right_ear = right_ear
            st.session_state.current_ear = ear
            cv2.drawContours(img, [left_eye], -1, (0, 255, 0), 1)
            cv2.drawContours(img, [right_eye], -1, (0, 255, 0), 1)
            if ear < EAR_THRESH:
                st.session_state.counter += 1
                if st.session_state.counter >= CONSEC_FRAMES:
                    st.session_state.alarm_on = True
                    overlay = img.copy()
                    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
                    text = "DROWSINESS DETECTED!"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 1.5, 3)[0]
                    text_x = (w - text_size[0]) // 2
                    text_y = (h + text_size[1]) // 2
                    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 3)
                    cv2.rectangle(img, (0, 0), (w, h), (0, 0, 255), 10)
            else:
                st.session_state.counter = 0
                st.session_state.alarm_on = False

        cv2.putText(img, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(img, f"L: {left_ear:.2f}  R: {right_ear:.2f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
        cv2.putText(img, f"Counter: {st.session_state.counter}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    def audio_frame_callback(frame):
        frame_sample_rate = frame.sample_rate
        frame_samples = frame.to_ndarray().shape[1]
        alarm_samples = st.session_state.alarm_array
        
        # Immediately play alarm sound when alarm_on is True
        if st.session_state.alarm_on and alarm_samples is not None and len(alarm_samples) > 0:
            # Repeat the alarm sound to fill the frame
            samples_needed = frame_samples
            repeats = (samples_needed // len(alarm_samples)) + 1
            out_samples = np.tile(alarm_samples, repeats)[:samples_needed]
            
            # Apply the user-set volume
            out_samples = (out_samples * ALARM_VOLUME).clip(-32768, 32767).astype(np.int16)
            
            # Format for output
            out_samples = out_samples.reshape(1, -1)
        else:
            out_samples = np.zeros((1, frame_samples), dtype=np.int16)
            
        out_frame = av.AudioFrame.from_ndarray(out_samples, format='s16', layout='mono')
        out_frame.sample_rate = frame_sample_rate
        return out_frame

    # --- Streamlit Layout ---
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### Live Drowsiness Detection")
        webrtc_streamer(
            key="drowsiness_detection",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            audio_frame_callback=audio_frame_callback,  # Enable audio output
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": True, "audio": False},  # Disable audio input but allow output
            video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=False, style={"width": "100%"}),
            audio_html_attrs=AudioHTMLAttributes(autoPlay=True, controls=True),
        )
    with col2:
        st.markdown("### Status")
        if st.session_state.alarm_on:
            st.error("⚠️ **DROWSINESS DETECTED!** ⚠️")
        else:
            st.success("Monitoring...")
        st.markdown(f"**Current EAR:** `{st.session_state.current_ear:.3f}`")
        st.markdown(f"**Left EAR:** `{st.session_state.left_ear:.3f}`")
        st.markdown(f"**Right EAR:** `{st.session_state.right_ear:.3f}`")
        st.markdown(f"**Counter:** `{st.session_state.counter}`")
        st.markdown(f"**Alarm Volume:** `{ALARM_VOLUME:.1f}x`")
        
        st.info("Yellow dots = landmarks. Green = eyes.")

    st.markdown("---")
    st.subheader("How to use")
    st.write("""
    1. Allow camera and microphone access.
    2. Position your face clearly in the camera.
    3. The system will detect if your eyes close for too long.
    4. An alarm will sound and visual warnings will appear if drowsiness is detected.
    5. Keep your volume up to hear the alarm.
    6. Adjust the alarm volume in the settings if needed.
    """)

    with st.expander("Troubleshooting"):
        st.write("""
        - **No face detected**: Make sure your face is clearly visible and well-lit.
        - **No sound**: Check that your system volume is up and browser audio is not muted.
        - **False alarms**: Adjust the EAR threshold in the settings.
        - **No detection**: Try refreshing the page or using a different browser.
        - **Audio issues**: If you can't hear the alarm, try increasing the alarm volume in settings.
        """)