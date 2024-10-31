import cv2
import dlib
import numpy as np
import time
import requests

# Load the pre-trained shape predictor model
PREDICTOR_PATH = '/home/amir-agho/programing/chet-detector/amin-code/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Variable to track previous head movement direction and last API request time
previous_head_movement = None
last_api_call_time = 0
api_call_interval = 2  # حداقل فاصله زمانی بین درخواست‌های API به ثانیه

# دیکشنری برای ذخیره تایمر هر جهت
head_movement_timers = {
    "Head turned left": None,
    "Head turned right": None,
    "Head tilted down": None
}

head_movement_hold_time = 3  # حداقل مدت زمان برای ارسال پیام

url = 'https://sngh4sn7-3000.euw.devtunnels.ms/api/direction'

# Function to send head movement to API
def send_head_movement_to_api(direction):
    data = {'direction': direction}
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print(f"Successfully sent {direction} to API")
        else:
            print(f"Failed to send {direction} to API, status code: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending data to API: {e}")

# Function to calculate the midpoint between two points
def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

# Function to detect head movement in 4 directions
def detect_head_movement(frame):
    global previous_head_movement  # Declare as global to keep track of changes across frames

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Calculate horizontal distance (left/right) for head movement
        left_eye = landmarks.part(36)
        right_eye = landmarks.part(45)
        nose = landmarks.part(30)

        eye_distance_x = right_eye.x - left_eye.x
        nose_offset_x = nose.x - (left_eye.x + right_eye.x) / 2

        # Calculate vertical distance (up/down) for head movement
        eye_average_y = (left_eye.y + right_eye.y) / 2
        nose_offset_y = nose.y - eye_average_y

        # Initialize head movement direction
        head_movement = "Head facing forward"

        # Detect horizontal head movement (left/right)
        if abs(nose_offset_x) > eye_distance_x * 0.2:  # Adjust sensitivity for left/right movement
            if nose_offset_x > 0:
                head_movement = "Head turned left"
            else:
                head_movement = "Head turned right"

        # Detect vertical head movement (up/down) with a neutral zone
        elif abs(nose_offset_y) > eye_distance_x * 0.55:  # Adjust sensitivity for up/down movement
            if nose_offset_y > 0:
                head_movement = "Head tilted down"

        # Check if the head movement direction has changed
        if head_movement in head_movement_timers:
            # اگر حرکت سر جدید است، تایمر شروع می‌شود
            if previous_head_movement != head_movement:
                head_movement_timers[head_movement] = time.time()  # شروع تایمر
            else:
                # اگر سر هنوز در همان جهت است، مدت زمان ماندن را محاسبه کن
                if time.time() - head_movement_timers[head_movement] > head_movement_hold_time:
                    send_head_movement_to_api(head_movement)  # ارسال به API
                    head_movement_timers[head_movement] = time.time()  # ریست تایمر بعد از ارسال پیام

        previous_head_movement = head_movement  # Update previous movement

        # Display the head movement on the frame
        cv2.putText(frame, head_movement, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

# Start capturing video from the laptop camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_head_movement(frame)
    cv2.imshow("Head Movement Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
