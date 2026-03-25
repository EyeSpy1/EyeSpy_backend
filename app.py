# app.py
import os
import io
import cv2
import dlib
import numpy as np
import pandas as pd

import sqlite3
import hashlib
import streamlit as st
from scipy.spatial import distance as dist
from pydub import AudioSegment
import base64
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import threading
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
def get_user_sessions_full(username):
    """Get recent sessions including session_data JSON (EAR history)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT session_date, total_time, drowsiness_events, avg_ear, min_ear, max_ear, session_data
        FROM user_sessions
        WHERE username = ?
        ORDER BY session_date DESC
        LIMIT 50
    """, (username,))
    rows = c.fetchall()
    conn.close()
    return rows

def get_session_ear_history(username, session_date):
    """Return list of EAR values for the given session timestamp."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT session_data FROM user_sessions
        WHERE username = ? AND session_date = ?
        LIMIT 1
    """, (username, session_date))
    row = c.fetchone()
    conn.close()
    if not row or not row[0]:
        return []
    try:
        return json.loads(row[0])
    except Exception:
        return []

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
# --- NEW: Adaptive threshold session defaults ---
for key, val in [
    ("adaptive_on", False),
    ("calib_seconds", 20),         # how long to learn your baseline
    ("drop_percent", 25),          # threshold = baseline_mean * (1 - drop_percent/100)
    ("min_adapt_thresh", 0.15),    # safety floor
    ("max_adapt_thresh", 0.35),    # safety ceiling
    ("baseline_buffer", []),       # collects EAR during calibration
    ("effective_ear_thresh", None),# computed adaptive threshold
    ("baseline_ready", False),     # becomes True after calibration
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

# --- WebRTC Setup ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class DrowsinessProcessor(VideoTransformerBase):
    def __init__(self):
        self.EAR_THRESH = 0.25
        self.CONSEC_FRAMES = 15
        self.adaptive_on = False
        self.calib_seconds = 20
        self.drop_percent = 25
        self.min_adapt_thresh = 0.15
        self.max_adapt_thresh = 0.35
        
        self.counter = 0
        self.alarm_on = False
        self.current_ear = 0.0
        self.left_ear = 0.0
        self.right_ear = 0.0
        self.drowsiness_count = 0
        self.session_start_time = time.time()
        
        self.ear_history = []
        self.baseline_buffer = []
        self.baseline_ready = False
        self.effective_ear_thresh = None
        self.baseline_dx = None
        self.baseline_dy = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if face_detection_ready:
            faces = detector(gray, 0)
            
            if not faces:
                self.counter = 0
                self.alarm_on = False
            
            for face in faces:
                shape = predictor(gray, face)
                shape_np = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
                
                for (x, y) in shape_np:
                    cv2.circle(img, (x, y), 2, (0, 255, 255), -1)
                
                left_eye = shape_np[42:48]
                right_eye = shape_np[36:42]
                
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
                
                self.left_ear = left_ear
                self.right_ear = right_ear
                self.current_ear = ear
                
                if self.adaptive_on:
                    elapsed = time.time() - self.session_start_time
                    if not self.baseline_ready:
                        if 0.05 < ear < 0.6:
                            self.baseline_buffer.append(ear)
                        if elapsed >= self.calib_seconds and len(self.baseline_buffer) >= 60:
                            baseline_mean = float(np.mean(self.baseline_buffer))
                            raw_thresh = baseline_mean * (1.0 - (self.drop_percent / 100.0))
                            clamped = max(self.min_adapt_thresh, min(self.max_adapt_thresh, raw_thresh))
                            self.effective_ear_thresh = clamped
                            self.baseline_ready = True

                self.ear_history.append(ear)
                if len(self.ear_history) > 1000:
                    self.ear_history = self.ear_history[-1000:]
                
                cv2.drawContours(img, [left_eye], -1, (0, 255, 0), 1)
                cv2.drawContours(img, [right_eye], -1, (0, 255, 0), 1)
                
                thr = self.effective_ear_thresh if (self.adaptive_on and self.baseline_ready and self.effective_ear_thresh) else self.EAR_THRESH
                
                if ear < thr:
                    self.counter += 1
                    if self.counter >= self.CONSEC_FRAMES:
                        if not self.alarm_on:
                            self.drowsiness_count += 1
                        self.alarm_on = True
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
                    self.counter = 0
                    self.alarm_on = False

                nose_point = (shape.part(30).x, shape.part(30).y)
                chin_point = (shape.part(8).x, shape.part(8).y)
                dx = nose_point[0] - chin_point[0]
                dy = nose_point[1] - chin_point[1]

                if self.baseline_dx is None or self.baseline_dy is None:
                    self.baseline_dx = dx
                    self.baseline_dy = dy

                dx = dx - self.baseline_dx
                dy = dy - self.baseline_dy

                if abs(dx) > 25:
                    cv2.putText(img, "DISTRACTION ALERT (Side View!)", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                elif dy < -20:
                    cv2.putText(img, "DISTRACTION ALERT (Looking Up!)", (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
                elif dy > 20:
                    cv2.putText(img, "DISTRACTION ALERT (Looking Down/Mobile!)", (50,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

            cv2.putText(img, f"EAR: {self.current_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            thr_disp = self.effective_ear_thresh if (self.adaptive_on and self.baseline_ready and self.effective_ear_thresh) else self.EAR_THRESH
            cv2.putText(img, f"THR: {thr_disp:.2f} ({'A' if (self.adaptive_on and self.baseline_ready) else 'S'})", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            cv2.putText(img, f"L: {self.left_ear:.2f}  R: {self.right_ear:.2f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
            cv2.putText(img, f"Counter: {self.counter}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            session_time = int(time.time() - self.session_start_time)
            cv2.putText(img, f"Time: {session_time}s  Events: {self.drowsiness_count}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

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

        st.write("---")
        st.subheader("Adaptive Threshold (Smart)")
        
        # 👇 YAHAN PE calibration duration daalo
        st.session_state.calib_seconds = st.number_input(
            "Calibration Duration (secs)",
            min_value=5, max_value=120,
            value=int(st.session_state.get("calib_seconds", 20)),
            step=1
        )

        st.session_state.adaptive_on = st.checkbox("Enable Adaptive EAR Threshold", value=st.session_state.adaptive_on)
        st.session_state.drop_percent = st.slider("Drop % from Baseline", 5, 50, int(st.session_state.drop_percent), 1)
        st.session_state.min_adapt_thresh = st.number_input("Min Threshold Clamp", 0.10, 0.30, float(st.session_state.min_adapt_thresh), 0.01)
        st.session_state.max_adapt_thresh = st.number_input("Max Threshold Clamp", 0.25, 0.50, float(st.session_state.max_adapt_thresh), 0.01)

        # Display the currently used threshold
        if st.session_state.adaptive_on and st.session_state.baseline_ready and st.session_state.effective_ear_thresh:
            st.success(f"Adaptive Threshold: {st.session_state.effective_ear_thresh:.3f}")
        elif st.session_state.adaptive_on:
            st.info("Adaptive Threshold: calibrating… (using slider until ready)")
        else:
            st.caption("Adaptive off: using slider value above.")

    # Main content tabs
    tab1, tab2 = st.tabs(["Live Detection", "Session History"])
    
    with tab1:
        st.markdown("### Live Drowsiness Detection (WebRTC Ready)")
        st.info("This uses your browser's webcam via WebRTC. It works perfectly on phones and laptops!")

        ctx = webrtc_streamer(
            key="drowsiness-detection",
            video_processor_factory=DrowsinessProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )

        if ctx.state.playing:
            st.success("🟢 Camera is running in your browser")
            if ctx.video_processor:
                # Sync UI settings to the processing thread
                ctx.video_processor.EAR_THRESH = EAR_THRESH
                ctx.video_processor.CONSEC_FRAMES = CONSEC_FRAMES
                ctx.video_processor.calib_seconds = st.session_state.calib_seconds
                ctx.video_processor.adaptive_on = st.session_state.adaptive_on
                ctx.video_processor.drop_percent = st.session_state.drop_percent
                ctx.video_processor.min_adapt_thresh = st.session_state.min_adapt_thresh
                ctx.video_processor.max_adapt_thresh = st.session_state.max_adapt_thresh

                # Save session logic
                st.markdown("---")
                if st.button("💾 Save Diagnostic Session to History"):
                    session_duration = time.time() - ctx.video_processor.session_start_time
                    avg_ear = sum(ctx.video_processor.ear_history) / len(ctx.video_processor.ear_history) if ctx.video_processor.ear_history else 0.0
                    min_ear = min(ctx.video_processor.ear_history) if ctx.video_processor.ear_history else 0.0
                    max_ear = max(ctx.video_processor.ear_history) if ctx.video_processor.ear_history else 0.0
                    
                    session_data = {
                        'session_date': datetime.now().isoformat(),
                        'total_time': int(session_duration),
                        'drowsiness_events': ctx.video_processor.drowsiness_count,
                        'avg_ear': avg_ear,
                        'min_ear': min_ear,
                        'max_ear': max_ear,
                        'ear_history': ctx.video_processor.ear_history[-100:]
                    }
                    save_session_data(st.session_state.username, session_data)
                    st.success(f"Session saved! Duration: {int(session_duration)}s, Events: {ctx.video_processor.drowsiness_count}")
        else:
            st.info("👆 Click 'START' on the video player above to begin monitoring.")

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
                # ---------- NEW: Export full history (summary) ----------
        st.write("---")
        st.markdown("#### Export History")

        # fetch extended rows including JSON to build a friendly CSV
        sessions_full = get_user_sessions_full(st.session_state.username)
        if sessions_full:
            # Build a flat table for export (JSON length only, not full timeseries)
            export_rows = []
            for (session_date, total_time, drowsiness_events, avg_ear, min_ear, max_ear, session_data) in sessions_full:
                try:
                    ear_list = json.loads(session_data) if session_data else []
                except Exception:
                    ear_list = []
                export_rows.append({
                    "session_date": session_date,
                    "total_time_secs": total_time,
                    "drowsiness_events": drowsiness_events,
                    "avg_ear": avg_ear,
                    "min_ear": min_ear,
                    "max_ear": max_ear,
                    "ear_points": len(ear_list)
                })
            df_export = pd.DataFrame(export_rows)

            csv_bytes = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download History (CSV)",
                data=csv_bytes,
                file_name=f"{st.session_state.username}_drowsiness_history.csv",
                mime="text/csv",
                help="Exports recent session summaries (not per-point EAR)."
            )
        else:
            st.info("No data to export yet.")
                # ---------- NEW: Trends & Analytics ----------
        st.write("---")
        st.markdown("### 📊 Session Trends & Analytics")

        sessions = get_user_sessions(st.session_state.username)
        if sessions:
            # Convert to DataFrame for easier plotting
            df_trend = pd.DataFrame(sessions, columns=[
                "session_date", "total_time", "drowsiness_events", "avg_ear", "min_ear", "max_ear"
            ])

            # Format datetime
            try:
                df_trend["session_date"] = pd.to_datetime(df_trend["session_date"])
            except Exception:
                pass  # keep raw strings if parsing fails

            # Show key trends
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Session Time", f"{df_trend['total_time'].mean():.1f}s")
            with col2:
                st.metric("Avg Events/Session", f"{df_trend['drowsiness_events'].mean():.2f}")
            with col3:
                st.metric("Best EAR", f"{df_trend['max_ear'].max():.3f}")

            # Trend 1: Drowsiness Events Over Time
            st.subheader("Drowsiness Events Over Time")
            st.line_chart(df_trend.set_index("session_date")["drowsiness_events"])

            # Trend 2: Average EAR Over Time
            st.subheader("Average EAR Over Time")
            st.line_chart(df_trend.set_index("session_date")["avg_ear"])

            # Trend 3: Session Duration Over Time
            st.subheader("Session Duration (secs) Over Time")
            st.bar_chart(df_trend.set_index("session_date")["total_time"])
        else:
            st.info("No session data available yet for analytics.")


        # ---------- NEW: Session-by-session graph & per-session export ----------
        st.write("---")
        st.markdown("#### Session-wise EAR Graph")

        if sessions_full:
            # A small selector to pick a session for plotting
            # Show most recent first; display a neat label
            def _fmt_label(sd, tt, ev):
                try:
                    dt = datetime.fromisoformat(sd)
                    dstr = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    dstr = sd
                return f"{dstr} | {tt}s | events: {ev}"

            session_labels = [
                _fmt_label(sd, tt, ev) for (sd, tt, ev, *_rest) in [(row[0], row[1], row[2]) for row in sessions_full]
            ]
            selected = st.selectbox("Pick a session to visualize", options=session_labels, index=0)

            # Map label back to session_date
            sel_idx = session_labels.index(selected)
            selected_session_date = sessions_full[sel_idx][0]

            # Fetch EAR history and plot
            ear_series = get_session_ear_history(st.session_state.username, selected_session_date)
            if ear_series:
                st.caption("EAR across frames (lower EAR ≈ eyes closed).")
                st.line_chart(ear_series)

                # Per-session export: full EAR timeseries
                per_rows = [{"frame_index": i, "ear": v} for i, v in enumerate(ear_series)]
                df_per = pd.DataFrame(per_rows)
                per_csv = df_per.to_csv(index=False).encode("utf-8")
                safe_dt = selected_session_date.replace(":", "-")
                st.download_button(
                    label="⬇️ Download this session's EAR (CSV)",
                    data=per_csv,
                    file_name=f"{st.session_state.username}_EAR_{safe_dt}.csv",
                    mime="text/csv",
                    help="Exports frame-wise EAR values for the selected session."
                )
            else:
                st.warning("No EAR history stored for this session.")
