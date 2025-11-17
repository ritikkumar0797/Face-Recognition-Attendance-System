# Face-Recognition-Attendance-System
Face Recognition Attendance System  A Streamlit-based app to mark student attendance using face recognition. Add new students via camera, take attendance with 8-hour interval restriction, and view live attendance. Real-time messages guide to call the next student after marking.

Face Recognition Attendance System

This project is a Streamlit-based face recognition attendance system designed to automate student attendance. It allows users to:

Add New Students – Capture student faces using a camera and save them for recognition.

Take Attendance – Automatically detect and recognize students in real-time using the webcam. Attendance can only be marked once every 8 hours to prevent duplicate entries. After marking, a message guides the user to call the next student.

View Live Attendance – Display a live table of attendance records, including student names, date, and time of entry.

The system uses OpenCV’s Haar cascade for face detection and a mean squared error (MSE) approach to recognize faces. All attendance records are saved in a CSV file, making it easy to manage and export.

This project is suitable for schools, colleges, or small organizations seeking a quick, automated attendance solution without manual logging. It ensures accuracy, reduces administrative work, and provides a user-friendly interface.
