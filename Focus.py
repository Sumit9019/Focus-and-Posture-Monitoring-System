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

# Function to detect a smartphone based on color (this can be improved with more advanced detection techniques)
def detect_smartphone(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of colors for detecting a smartphone (adjust these values as needed)
    lower_color = np.array([100, 100, 100])  # Example lower bound
    upper_color = np.array([140, 255, 255])  # Example upper bound

    # Create a mask for the specified color range
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Check if there are any non-zero pixels in the mask
    if cv2.countNonZero(mask) > 500:  # Adjust the threshold as needed
        return True  # Smartphone detected
    return False  # No smartphone detected

# Drowsiness detection thresholds
thresh = 0.25
frame_check = 20

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

        # Draw contours around the eyes and mouth
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

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

        # Face rotation detection
        if face_orientation(nose, chin, left_face, right_face):
            cv2.putText(frame, "****************ALERT! FACE ROTATED!****************", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            play_warning("Warning! Your face is not looking straight. Please focus on the road.")

        # Posture detection (slouching or improper sitting)
        if check_posture(nose, chin):
            cv2.putText(frame, "****************ALERT! IMPROPER POSTURE!****************", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            play_warning("Warning! You are not sitting properly. Please sit straight.")

        # Smartphone detection
        if detect_smartphone(frame):
            cv2.putText(frame, "****************ALERT! SMARTPHONE DETECTED!****************", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            play_warning("Warning! Smartphone detected. Please focus on the road.")

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break loop if "q" is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Clean up resources
cv2.destroyAllWindows()
cap.release()
