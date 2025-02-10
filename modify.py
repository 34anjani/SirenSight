from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import time

# Initialize pygame mixer for sound
mixer.init()
mixer.music.load("C:/Users/Lenovo/OneDrive/Desktop/DDS project/music.wav")

# Define paths for alerts
first_alert_path = "C:/Users/Lenovo/OneDrive/Desktop/DDS project/music.wav"
second_alert_path = "C:/Users/Lenovo/OneDrive/Desktop/DDS project/mixkit-ambulance-siren-us-1642.wav"

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

thresh = 0.25
frame_check = 20

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("C:/Users/Lenovo/OneDrive/Desktop/DDS project/shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

cap = cv2.VideoCapture(0)
flag = 0
start_time = None
first_alert_time = None
alert_triggered = False

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
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

        if ear < thresh:
            if start_time is None:
                start_time = time.time()  # Start timer when eye is first closed

            elapsed_time = time.time() - start_time

            if elapsed_time >= 2.5 and first_alert_time is None:  # First alert after 2.5 seconds
                mixer.music.load(first_alert_path)
                mixer.music.play()
                first_alert_time = time.time()
                cv2.putText(frame, "****************FIRST ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************FIRST ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if first_alert_time and (time.time() - first_alert_time >= 2.5):  # Second alert after 2.5 seconds of first
                if not alert_triggered:
                    mixer.music.load(second_alert_path)
                    mixer.music.play()
                    alert_triggered = True

                cv2.putText(frame, "****************SECOND ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************SECOND ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            start_time = None
            first_alert_time = None
            alert_triggered = False
            mixer.music.stop()

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
