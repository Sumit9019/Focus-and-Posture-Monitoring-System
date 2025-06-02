from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import pyttsx3
import time
from pygame import mixer
import numpy as np

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Initialize the mixer for playing sounds
mixer.init()
mixer.music.load("music.wav")  # Load your beep sound file

# Function to play beep sound and then the warning
def play_warning(text):
    # Play the beep sound
    mixer.music.play()
    time.sleep(1)  # Wait for 1 second (adjust as needed based on the beep sound length)
    
    # After the beep, speak the warning
    engine.say(text)
    engine.runAndWait()

# Function to calculate eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate mouth aspect ratio (MAR) for yawning detection
def mouth_aspect_ratio(mouth):
    # Vertical distances
    A = distance.euclidean(mouth[13], mouth[19])  # 51, 59
    B = distance.euclidean(mouth[15], mouth[17])  # 53, 57
    # Horizontal distance
    C = distance.euclidean(mouth[12], mouth[16])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

# Function to check face orientation based on landmarks
def face_orientation(nose_point, chin_point, left_point, right_point):
    nose_to_left = distance.euclidean(nose_point, left_point)
    nose_to_right = distance.euclidean(nose_point, right_point)
    if abs(nose_to_left - nose_to_right) > 50:  # Threshold to detect face rotation
        return True
    return False

# Function to check if the driver is slouching or lying down
def check_posture(nose_point, chin_point):
    vertical_diff = abs(nose_point[1] - chin_point[1])
    if vertical_diff > 80:  # Threshold to detect improper posture
        return True
    return False

# Drowsiness detection thresholds
thresh = 0.25
frame_check = 20

# Yawning detection thresholds
mar_thresh = 0.7  # Adjust as needed
mar_frame_check = 15
mar_flag = 0

# Load facial landmarks and initialize the detector
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Facial landmarks for eyes, nose, and posture detection
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mouthStart, mouthEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]  # Mouth landmarks
(noseIdx, chinIdx) = (33, 8)  # Nose and chin landmarks
(leftIdx, rightIdx) = (0, 16)  # Leftmost and rightmost face landmarks

cap = cv2.VideoCapture(0)
flag = 0
# Yawning detection variables
YAWN_THRESH = 0.7  # Adjust as needed
YAWN_CONSEC_FRAMES = 15
yawn_counter = 0

while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # Get the landmarks for eyes, nose, chin, mouth, and face orientation points
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mouthStart:mouthEnd]  # Get mouth landmarks
        nose = shape[noseIdx]
        chin = shape[chinIdx]
        left_face = shape[leftIdx]
        right_face = shape[rightIdx]

        # Calculate EAR for drowsiness detection
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Calculate MAR for yawning detection
        mar = mouth_aspect_ratio(mouth)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (255, 0, 0), 1)

        # Drowsiness detection
        if ear < thresh:
            flag += 1
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                play_warning("Warning! You are feeling drowsy. Please pay attention.")
        else:
            flag = 0

        # Yawning detection
        if mar > YAWN_THRESH:
            yawn_counter += 1
            if yawn_counter >= YAWN_CONSEC_FRAMES:
                cv2.putText(frame, "****************YAWNING DETECTED!****************", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                play_warning("Warning! You are yawning. Please stay alert.")
        else:
            yawn_counter = 0

        # Face rotation detection
        if face_orientation(nose, chin, left_face, right_face):
            cv2.putText(frame, "****************ALERT! FACE ROTATED!****************", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            play_warning("Warning! Your face is not looking straight. Please focus on the road.")

        # Posture detection (slouching or improper sitting)
        if check_posture(nose, chin):
            cv2.putText(frame, "****************ALERT! IMPROPER POSTURE!****************", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            play_warning("Warning! You are yawning. Please stay alert.")

        # Smartphone detection
        # if detect_smartphone(frame):
        #     cv2.putText(frame, "****************ALERT! SMARTPHONE DETECTED!****************", (10, 120),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #     play_warning("Warning! Smartphone detected. Please focus on the road.")

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break loop if "q" is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Clean up resources
cv2.destroyAllWindows()
cap.release()

