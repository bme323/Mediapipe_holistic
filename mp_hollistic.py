import cv2
from picamera2 import Picamera2

import mediapipe as mp 
import time

import numpy as np

mp_holistic = mp.solutions.holistic 
mp_pose = mp.solutions.pose 
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# input = 1

def PoseEstimation():
    # Get webcam input
    # cap = cv2.VideoCapture(input)
    
    piCam=Picamera2()
    piCam.preview_configuration.main.size=(1280,720)
    piCam.preview_configuration.main.format="RGB888"
    piCam.preview_configuration.align()
    piCam.configure("preview")
    piCam.start()

    #`Initialise time and fps variables 
    time_start = 0
    fps = 0
    frame = 0
    landmark_list = []
    av_visibility = []
    fps_array = []

    # Begin new instance of mediapipe feed
    with mp_holistic.Holistic(
        model_complexity = 1,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5) as holistic:

        while True:

            '''    
            isTrue, image = cap.read() 
            

            if not isTrue:
                print("Empty camera frame")
                continue 
            '''
            image = piCam.capture_array()
            # Improve performance 
            image = cv2.resize(image, (1280,720))
            # Recolour image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Make detection and store in output array
            output = holistic.process(image)
            image.flags.writeable = True
            # Recolour to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            h,w,c = image.shape
            
            # Draw landmarks on image
            mp_draw.draw_landmarks(
                image,
                output.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()) # connect landmarks
            
            mp_draw.draw_landmarks(
                image,
                output.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()) # connect landmarks

            mp_draw.draw_landmarks(
                image,
                output.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()) # connect landmarks


            # Display 
            # Mirror image for webcam display
            image = cv2.flip(image,1)

            # Calculate fps
            time_end = time.time()
            dt = time_end - time_start
            fps = 1/dt
            time_start = time_end
            # Draw fps onto image
            show_fps = "FPS: {:.3} ".format(fps)

            fps_array.append(fps)

            cv2.putText(image, show_fps, (5, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            
            # Visualise image and flip display
            cv2.imshow('Video', image)

            # break loop and close windows if q key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    return output

PoseEstimation()
