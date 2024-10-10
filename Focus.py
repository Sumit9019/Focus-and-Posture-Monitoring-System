# from scipy.spatial import distance
# from imutils import face_utils
# import imutils
# import dlib
# import cv2
# import pyttsx3
# import time
# from pygame import mixer
# import numpy as np
# from geopy.distance import geodesic
# from geopy.geocoders import Nominatim

# # Initialize the text-to-speech engine
# engine = pyttsx3.init()

# # Initialize the mixer for playing sounds
# mixer.init()
# mixer.music.load("music.wav")  # Load your beep sound file

# # Function to play beep sound and then the warning
# def play_warning(text):
#     # Play the beep sound
#     mixer.music.play()
#     time.sleep(1)  # Wait for 1 second (adjust as needed based on the beep sound length)
    
#     # After the beep, speak the warning
#     engine.say(text)
#     engine.runAndWait()

# # Function to calculate eye aspect ratio
# def eye_aspect_ratio(eye):
#     A = distance.euclidean(eye[1], eye[5])
#     B = distance.euclidean(eye[2], eye[4])
#     C = distance.euclidean(eye[0], eye[3])
#     ear = (A + B) / (2.0 * C)
#     return ear

# # Function to check face orientation based on landmarks
# def face_orientation(nose_point, chin_point, left_point, right_point):
#     nose_to_left = distance.euclidean(nose_point, left_point)
#     nose_to_right = distance.euclidean(nose_point, right_point)
#     if abs(nose_to_left - nose_to_right) > 50:  # Threshold to detect face rotation
#         return True
#     return False

# # Drowsiness detection thresholds
# thresh = 0.25
# frame_check = 20

# # Load facial landmarks and initialize the detector
# detect = dlib.get_frontal_face_detector()
# predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Facial landmarks for eyes, nose, and face orientation detection
# (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
# (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
# (noseIdx, chinIdx) = (33, 8)  # Nose and chin landmarks
# (leftIdx, rightIdx) = (0, 16)  # Leftmost and rightmost face landmarks

# # Get user name
# user_name = input("Please enter your name: ")
# engine.say(f"Hello {user_name}, the driving alertness system is starting.")
# engine.runAndWait()  # Speak the user's name

# # Get source and destination for travel
# geolocator = Nominatim(user_agent="geoapiExercises")
# source_city = input("Please enter the source city: ")
# destination_city = input("Please enter the destination city: ")

# source_location = geolocator.geocode(source_city)
# destination_location = geolocator.geocode(destination_city)

# if source_location and destination_location:
#     source_coords = (source_location.latitude, source_location.longitude)
#     destination_coords = (destination_location.latitude, destination_location.longitude)
#     distance_km = geodesic(source_coords, destination_coords).kilometers
    
#     # Calculate recommended driving and rest times
#     driving_time_hours = distance_km / 60  # Assuming an average driving speed of 60 km/h
#     recommended_break_time = driving_time_hours / 2  # Suggest breaks every 2 hours

#     engine.say(f"The distance from {source_city} to {destination_city} is {distance_km:.2f} kilometers.")
#     engine.say(f"You should take breaks after every 2 hours of driving.")
#     engine.say(f"Estimated time to reach your destination is {driving_time_hours:.2f} hours including breaks.")
#     engine.runAndWait()

# else:
#     engine.say("Sorry, I couldn't find the locations.")
#     engine.runAndWait()

# # Start capturing video
# cap = cv2.VideoCapture(0)
# flag = 0
# smartphone_flag = 0
# smartphone_time = 0  # Time counter for smartphone usage detection
# smartphone_visible_time = 0  # Time counter for smartphone visibility detection

# while True:
#     ret, frame = cap.read()
#     frame = imutils.resize(frame, width=450)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     subjects = detect(gray, 0)

#     # Define a region of interest (ROI) for smartphone detection
#     roi = frame[300:400, 100:300]  # Adjust these coordinates based on camera view
#     roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

#     # Define color range for smartphone detection (example: dark colors)
#     lower_color = np.array([0, 0, 0])   # Lower bound for dark colors in HSV
#     upper_color = np.array([180, 255, 70])  # Upper bound for dark colors in HSV

#     # Create a mask for detecting dark colors
#     mask = cv2.inRange(roi_hsv, lower_color, upper_color)
#     dark_area_percentage = np.sum(mask) / (mask.size)  # Calculate the percentage of dark area

#     for subject in subjects:
#         shape = predict(gray, subject)
#         shape = face_utils.shape_to_np(shape)

#         # Get the landmarks for eyes, nose, chin, and face orientation points
#         leftEye = shape[lStart:lEnd]
#         rightEye = shape[rStart:rEnd]
#         nose = shape[noseIdx]
#         chin = shape[chinIdx]
#         left_face = shape[leftIdx]
#         right_face = shape[rightIdx]

#         # Calculate EAR for drowsiness detection
#         leftEAR = eye_aspect_ratio(leftEye)
#         rightEAR = eye_aspect_ratio(rightEye)
#         ear = (leftEAR + rightEAR) / 2.0

#         # Draw contours around the eyes
#         leftEyeHull = cv2.convexHull(leftEye)
#         rightEyeHull = cv2.convexHull(rightEye)
#         cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
#         cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

#         # Drowsiness detection
#         if ear < thresh:
#             flag += 1
#             if flag >= frame_check:
#                 cv2.putText(frame, "****************ALERT!****************", (10, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                 cv2.putText(frame, "****************ALERT!****************", (10, 325),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                 play_warning(f"{user_name}, Warning! You are feeling drowsy. Please pay attention.")
#         else:
#             flag = 0

#         # Face rotation detection
#         if face_orientation(nose, chin, left_face, right_face):
#             cv2.putText(frame, "****************ALERT! FACE ROTATED!****************", (10, 60),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#             play_warning(f"{user_name}, Warning! Your face is not looking straight. Please focus on the road.")

#         # Smartphone usage detection (looking down)
#         if nose[1] > chin[1]:  # Simple check if the nose is below the chin
#             smartphone_flag += 1
#             if smartphone_flag >= 50:  # Assuming 1 frame ~ 0.1 seconds, for 5 seconds = 50 frames
#                 smartphone_time += 1
#                 if smartphone_time >= 5:  # If looking down for 5 seconds
#                     cv2.putText(frame, "****************ALERT! USING SMARTPHONE!****************", (10, 120),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                     play_warning(f"{user_name}, Warning! You have been using your smartphone for too long.")
#         else:
#             smartphone_flag = 0
#             smartphone_time = 0  # Reset time if not looking down

#         # Smartphone visibility detection
#         if dark_area_percentage > 0.3:  # Threshold for smartphone detection (30% of ROI is dark)
#             smartphone_visible_time += 1
#             if smartphone_visible_time >= 50:  # If smartphone is visible for more than 5 seconds
#                 cv2.putText(frame, "****************ALERT! SMARTPHONE VISIBLE!****************", (10, 150),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                 play_warning(f"{user_name}, Warning! Your smartphone is visible. Please focus on the road.")
#         else:
#             smartphone_visible_time = 0  # Reset if smartphone is not visible

#     # Display the frame
#     cv2.imshow("Frame", frame)

#     # Break loop if "q" is pressed
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         break

# # Clean up resources
# cv2.destroyAllWindows()
# cap.release()

# NEW CODE STARTS HERE
from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import pyttsx3
import time
from pygame import mixer

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

# Drowsiness detection thresholds
thresh = 0.25
frame_check = 20

# Load facial landmarks and initialize the detector
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Facial landmarks for eyes, nose, and posture detection
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
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

        # Get the landmarks for eyes, nose, chin, and face orientation points
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        nose = shape[noseIdx]
        chin = shape[chinIdx]
        left_face = shape[leftIdx]
        right_face = shape[rightIdx]

        # Calculate EAR for drowsiness detection
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw contours around the eyes
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

    # Display the frame
    cv2.imshow("Frame", frame)

    # Break loop if "q" is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Clean up resources
cv2.destroyAllWindows()
cap.release()



