# app.py
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
import base64
import time
import json
from datetime import datetime

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
    # Add user_sessions table for storing session data
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            session_date TEXT,
            total_time INTEGER,
            drowsiness_events INTEGER,
            avg_ear REAL,
            min_ear REAL,
            max_ear REAL,
            session_data TEXT,
            FOREIGN KEY (username) REFERENCES users (username)
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

def save_session_data(username, session_data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO user_sessions 
        (username, session_date, total_time, drowsiness_events, avg_ear, min_ear, max_ear, session_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        username,
        session_data['session_date'],
        session_data['total_time'],
        session_data['drowsiness_events'],
        session_data['avg_ear'],
        session_data['min_ear'],
        session_data['max_ear'],
        json.dumps(session_data['ear_history'])
    ))
    conn.commit()
    conn.close()

def get_user_sessions(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT session_date, total_time, drowsiness_events, avg_ear, min_ear, max_ear
        FROM user_sessions 
        WHERE username = ?
        ORDER BY session_date DESC
        LIMIT 10
    """, (username,))
    rows = c.fetchall()
    conn.close()
    return rows

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
except Exception:
    face_detection_ready = False

# --- EAR Calculation ---
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C) if C else 0

# --- Session state defaults ---
for key, val in [
    ("authenticated", False), ("username", ""), ("counter", 0), ("alarm_on", False),
    ("debug", False), ("ear_values", []), ("alarm_array", None), ("camera_running", False),
    ("current_ear", 0.0), ("left_ear", 0.0), ("right_ear", 0.0), ("custom_audio_b64", None),
    ("session_start_time", None), ("drowsiness_count", 0), ("ear_history", [])
]:
    if key not in st.session_state:
        st.session_state[key] = val

# --- helper: generate beep wav base64 once (for browser embed) ---
def create_beep_wavbase64(duration_secs=0.8, frequency=880, sample_rate=44100, volume=0.8):
    t = np.linspace(0, duration_secs, int(sample_rate * duration_secs), False)
    beep = np.sin(2 * np.pi * frequency * t) * 0.5
    beep += np.sin(2 * np.pi * frequency * 1.5 * t) * 0.3
    beep += np.sin(2 * np.pi * frequency * 2.0 * t) * 0.15
    arr = (beep * 32767).astype(np.int16)
    raw_bytes = arr.tobytes()
    audio_seg = AudioSegment(
        data=raw_bytes,
        sample_width=2,
        frame_rate=sample_rate,
        channels=1
    )
    buf = io.BytesIO()
    audio_seg.export(buf, format="wav")
    wav_bytes = buf.getvalue()
    b64 = base64.b64encode(wav_bytes).decode("utf-8")
    return b64

def audio_file_to_base64(audio_file):
    try:
        audio_seg = AudioSegment.from_file(audio_file)
        buf = io.BytesIO()
        audio_seg.export(buf, format="wav")
        wav_bytes = buf.getvalue()
        b64 = base64.b64encode(wav_bytes).decode("utf-8")
        return b64
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None

_beep_b64 = create_beep_wavbase64()

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
        st.error("Face detection model not found or corrupted. Put 'shape_predictor_68_face_landmarks.dat' in the 'models' folder.")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        EAR_THRESH = st.slider("Eye Aspect Ratio Threshold", 0.15, 0.40, 0.25, 0.01)
        CONSEC_FRAMES = st.slider("Consecutive Frames", 5, 30, 15, 1)
        
        st.write("---")
        st.subheader("Custom Audio")
        uploaded_audio = st.file_uploader("Upload Custom Alarm Audio", type=['wav', 'mp3', 'ogg', 'flac'])
        if uploaded_audio is not None:
            if st.button("Process Audio"):
                custom_b64 = audio_file_to_base64(uploaded_audio)
                if custom_b64:
                    st.session_state.custom_audio_b64 = custom_b64
                    st.success("Custom audio uploaded successfully!")
        
        if st.session_state.custom_audio_b64:
            if st.button("Reset to Default Audio"):
                st.session_state.custom_audio_b64 = None
                st.success("Reset to default beep sound!")
        
        st.write("---")
        if st.button("Log Out"):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.session_state.camera_running = False
            st.experimental_rerun()

    # Main content tabs
    tab1, tab2 = st.tabs(["Live Detection", "Session History"])
    
    with tab1:
        st.markdown("### Live Drowsiness Detection (OpenCV fallback)")
        st.info("This uses your local webcam via OpenCV. Click 'Start Camera & Detection' then allow camera access.")

        # Show current status
        if st.session_state.camera_running:
            st.success("🟢 Camera is running")
        else:
            st.info("🔴 Camera is stopped")

        # Controls
        col1, col2 = st.columns(2)
        with col1:
            if not st.session_state.camera_running:
                if st.button("🎥 Start Camera & Detection", key="start_btn"):
                    st.session_state.camera_running = True
                    st.session_state.session_start_time = time.time()
                    st.session_state.drowsiness_count = 0
                    st.session_state.ear_history = []
                    st.experimental_rerun()
            else:
                st.button("🎥 Start Camera & Detection", disabled=True)
        
        with col2:
            if st.session_state.camera_running:
                if st.button("⏹️ Stop Camera", key="stop_btn"):
                    st.session_state.camera_running = False
                    
                    # Save session data when stopping
                    if st.session_state.session_start_time:
                        session_duration = time.time() - st.session_state.session_start_time
                        if st.session_state.ear_history:
                            avg_ear = sum(st.session_state.ear_history) / len(st.session_state.ear_history)
                            min_ear = min(st.session_state.ear_history)
                            max_ear = max(st.session_state.ear_history)
                        else:
                            avg_ear = min_ear = max_ear = 0.0
                        
                        session_data = {
                            'session_date': datetime.now().isoformat(),
                            'total_time': int(session_duration),
                            'drowsiness_events': st.session_state.drowsiness_count,
                            'avg_ear': avg_ear,
                            'min_ear': min_ear,
                            'max_ear': max_ear,
                            'ear_history': st.session_state.ear_history[-100:]  # Keep last 100 readings
                        }
                        save_session_data(st.session_state.username, session_data)
                        st.success(f"Session saved! Duration: {int(session_duration)}s, Drowsiness events: {st.session_state.drowsiness_count}")
                    
                    # Reset session variables
                    st.session_state.session_start_time = None
                    st.session_state.drowsiness_count = 0
                    st.session_state.ear_history = []
                    st.experimental_rerun()
            else:
                st.button("⏹️ Stop Camera", disabled=True)

        # Main camera loop
        if st.session_state.camera_running:
            st.info("🎥 Initializing camera...")
            
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Cannot open camera (index 0). Close other apps using camera or try a different camera index.")
                st.session_state.camera_running = False
                st.experimental_rerun()
            else:
                st.success("✅ Camera connected successfully!")
                
                try:
                    # placeholders (create ONCE outside loop)
                    img_placeholder = st.empty()
                    status1 = st.empty() 
                    status2 = st.empty()
                    audio_placeholder = st.empty()

                    counter = 0
                    prev_alarm = False
                    frame_count = 0

                    while st.session_state.camera_running:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("❌ Can't receive frame (stream end?). Stopping camera.")
                            st.session_state.camera_running = False
                            break

                        frame_count += 1
                        
                        # detection
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        h, w = frame.shape[:2]
                        faces = detector(gray, 0)
                        ear = 0.0
                        left_ear = 0.0
                        right_ear = 0.0

                        if not faces:
                            counter = 0
                            st.session_state.alarm_on = False
                        
                        for face in faces:
                            shape = predictor(gray, face)
                            shape_np = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
                            
                            # Draw face landmarks
                            for (x, y) in shape_np:
                                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                            
                            # Extract eye regions
                            left_eye = shape_np[42:48]
                            right_eye = shape_np[36:42]
                            
                            # Calculate EAR
                            left_ear = eye_aspect_ratio(left_eye)
                            right_ear = eye_aspect_ratio(right_eye)
                            ear = (left_ear + right_ear) / 2.0
                            
                            # Update session state
                            st.session_state.left_ear = left_ear
                            st.session_state.right_ear = right_ear
                            st.session_state.current_ear = ear
                            
                            # Store EAR history for analytics
                            st.session_state.ear_history.append(ear)
                            if len(st.session_state.ear_history) > 1000:  # Keep only last 1000 readings
                                st.session_state.ear_history = st.session_state.ear_history[-1000:]
                            
                            # Draw eye contours
                            cv2.drawContours(frame, [left_eye], -1, (0, 255, 0), 1)
                            cv2.drawContours(frame, [right_eye], -1, (0, 255, 0), 1)
                            
                            # Drowsiness detection
                            if ear < EAR_THRESH:
                                counter += 1
                                if counter >= CONSEC_FRAMES:
                                    if not st.session_state.alarm_on:  # New drowsiness event
                                        st.session_state.drowsiness_count += 1
                                    st.session_state.alarm_on = True
                                    
                                    # Create red overlay for drowsiness
                                    overlay = frame.copy()
                                    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                                    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
                                    
                                    # Add warning text
                                    text = "DROWSINESS DETECTED!"
                                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 1.5, 3)[0]
                                    text_x = (w - text_size[0]) // 2
                                    text_y = (h + text_size[1]) // 2
                                    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 3)
                                    cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
                            else:
                                counter = 0
                                st.session_state.alarm_on = False

                        # Draw statistics on frame
                        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        cv2.putText(frame, f"L: {left_ear:.2f}  R: {right_ear:.2f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
                        cv2.putText(frame, f"Counter: {counter}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        
                        # Show session info
                        if st.session_state.session_start_time:
                            session_time = int(time.time() - st.session_state.session_start_time)
                            cv2.putText(frame, f"Time: {session_time}s  Events: {st.session_state.drowsiness_count}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        else:
                            session_time = 0

                        # Update display (no reflow)
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img_placeholder.image(img_rgb, use_column_width=True)

                        # Update status
                        if st.session_state.alarm_on:
                            status1.error("⚠️ DROWSINESS DETECTED! ⚠️")
                        else:
                            status1.success("✅ Monitoring...")

                        status2.markdown(f"**EAR:** `{st.session_state.current_ear:.3f}` **L:** `{st.session_state.left_ear:.3f}` **R:** `{st.session_state.right_ear:.3f}` | **Time:** {session_time}s **Events:** {st.session_state.drowsiness_count}")

                        # Handle audio (only on state change to prevent page jump)
                        if st.session_state.alarm_on != prev_alarm:
                            if st.session_state.alarm_on:
                                # Use custom audio if available, otherwise default beep
                                audio_b64 = st.session_state.custom_audio_b64 if st.session_state.custom_audio_b64 else _beep_b64
                                audio_html = f"""
                                <audio autoplay loop>
                                  <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
                                  Your browser does not support the audio element.
                                </audio>
                                """
                                audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
                            else:
                                audio_placeholder.empty()
                            prev_alarm = st.session_state.alarm_on

                        # Control frame rate (~30 FPS)
                        time.sleep(0.03)

                except Exception as e:
                    st.error(f"❌ Camera loop error: {e}")
                    st.session_state.camera_running = False
                finally:
                    cap.release()
                    cv2.destroyAllWindows()
                    if st.session_state.camera_running:
                        st.session_state.camera_running = False
                        st.experimental_rerun()

        else:
            st.info("👆 Click 'Start Camera & Detection' to begin monitoring")

    with tab2:
        st.markdown("### Session History")
        sessions = get_user_sessions(st.session_state.username)
        
        if sessions:
            st.markdown("#### Recent Sessions")
            for i, session in enumerate(sessions):
                session_date, total_time, drowsiness_events, avg_ear, min_ear, max_ear = session
                
                # Parse datetime for better display
                try:
                    dt = datetime.fromisoformat(session_date)
                    formatted_date = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    formatted_date = session_date
                
                with st.expander(f"Session {i+1}: {formatted_date}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Duration", f"{total_time}s")
                        st.metric("Drowsiness Events", drowsiness_events)
                    with col2:
                        st.metric("Average EAR", f"{avg_ear:.3f}")
                        st.metric("Min EAR", f"{min_ear:.3f}")
                    with col3:
                        st.metric("Max EAR", f"{max_ear:.3f}")
                        if drowsiness_events > 0:
                            st.error("⚠️ Drowsiness Detected")
                        else:
                            st.success("✅ Good Session")
        else:
            st.info("No session data available. Start a detection session to see your history!")