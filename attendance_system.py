import streamlit as st 
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# ------------------- SETTINGS -------------------
st.set_page_config(page_title="Face Recognition Attendance", layout="wide")

# Directories & files
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(ATTENDANCE_FILE, index=False)

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ------------------- FUNCTIONS -------------------
def save_attendance(name):
    """Save attendance if 8 hours have passed since last attendance for the same student"""
    df = pd.read_csv(ATTENDANCE_FILE)
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    
    # Filter for this student
    student_df = df[df["Name"] == name]
    if not student_df.empty:
        # Get last attendance time
        last_time_str = student_df.iloc[-1]["Time"]
        last_date_str = student_df.iloc[-1]["Date"]
        last_datetime = datetime.strptime(f"{last_date_str} {last_time_str}", "%Y-%m-%d %H:%M:%S")
        if now - last_datetime < timedelta(hours=8):
            return False  # Less than 8 hours, cannot mark again

    # Mark attendance
    df = pd.concat([df, pd.DataFrame([[name, date, time]], columns=["Name","Date","Time"])])
    df.to_csv(ATTENDANCE_FILE, index=False)
    return True

def load_known_faces():
    """Load known faces as grayscale images"""
    known_faces = {}
    for file in os.listdir(KNOWN_FACES_DIR):
        if file.endswith(".jpg") or file.endswith(".png"):
            name = os.path.splitext(file)[0]
            img = cv2.imread(os.path.join(KNOWN_FACES_DIR, file), cv2.IMREAD_GRAYSCALE)
            known_faces[name] = img
    return known_faces

def recognize_face(face_roi, known_faces, threshold=2000):
    """Recognize face using mean squared error"""
    name_detected = "Unknown"
    min_dist = float("inf")
    for name, known_face in known_faces.items():
        resized_face = cv2.resize(face_roi, (known_face.shape[1], known_face.shape[0]))
        mse = np.mean((resized_face - known_face) ** 2)
        if mse < min_dist and mse < threshold:
            min_dist = mse
            name_detected = name
    return name_detected

# ------------------- STREAMLIT SIDEBAR -------------------
mode = st.sidebar.selectbox("Choose Action", ["Add New Student", "Take Attendance", "Live Attendance List"])
st.title(f"Mode: {mode}")

camera_placeholder = st.empty()
message_placeholder = st.empty()
live_attendance_placeholder = st.empty()

# ------------------- ADD NEW STUDENT -------------------
if mode == "Add New Student":
    st.subheader("Add a New Student")
    student_name = st.text_input("Student Name")
    img_file_buffer = st.camera_input("Capture Student Face")

    if student_name and img_file_buffer:
        bytes_data = img_file_buffer.getvalue()
        nparr = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(KNOWN_FACES_DIR, f"{student_name}.jpg"), img)
        st.success(f"Saved face for {student_name} ✅")

# ------------------- TAKE ATTENDANCE -------------------
elif mode == "Take Attendance":
    st.subheader("Take Attendance")
    known_faces = load_known_faces()
    start_cam = st.button("Start Camera")
    stop_cam = st.button("Stop Camera")
    
    if start_cam:
        cap = cv2.VideoCapture(0)
        st.session_state['running'] = True

        while st.session_state.get('running', False):
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access camera")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

            for (x, y, w, h) in faces:
                face_roi = gray_frame[y:y+h, x:x+w]
                name = recognize_face(face_roi, known_faces)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                if name != "Unknown":
                    if save_attendance(name):
                        message_placeholder.success(f"Attendance marked for {name} ✅\nCall next student.")
                    else:
                        message_placeholder.warning(f"Cannot mark {name} again within 8 hours ⏰")

            camera_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            if stop_cam:
                st.session_state['running'] = False
                break

        cap.release()
        camera_placeholder.empty()

# ------------------- LIVE ATTENDANCE LIST -------------------
elif mode == "Live Attendance List":
    st.subheader("Live Attendance Table")
    df = pd.read_csv(ATTENDANCE_FILE)
    live_attendance_placeholder.dataframe(df)
