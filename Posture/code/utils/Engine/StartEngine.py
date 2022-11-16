import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import numpy as np
import csv
import pickle 
import pandas as pd
import warnings
import math

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def EvalDetect(image,landmark_pose):
  image_height, image_width, _ = image.shape 
  detection = True

  RIGHT_SHOULDER = landmark_pose[12]
  RIGHT_SHOULDER_X = int(RIGHT_SHOULDER.x * image_width)
  RIGHT_SHOULDER_Y = int(RIGHT_SHOULDER.y * image_height)
  if (RIGHT_SHOULDER.visibility < 0.5):
    detection = False

  LEFT_SHOULDER = landmark_pose[11]
  LEFT_SHOULDER_X = int(LEFT_SHOULDER.x * image_width)
  LEFT_SHOULDER_Y = int(LEFT_SHOULDER.y * image_height)
  if (LEFT_SHOULDER.visibility < 0.5):
    detection = False

  CENTER_SHOULDER_X = int((RIGHT_SHOULDER_X+LEFT_SHOULDER_X)/2)
  CENTER_SHOULDER_Y = int((RIGHT_SHOULDER_Y+LEFT_SHOULDER_Y)/2) 

  RIGHT_ELBOW = landmark_pose[14]
  RIGHT_ELBOW_X = int(RIGHT_ELBOW.x * image_width)
  RIGHT_ELBOW_Y = int(RIGHT_ELBOW.y * image_height)
  if (RIGHT_ELBOW.visibility < 0.5):
    detection = False

  LEFT_ELBOW = landmark_pose[13]
  LEFT_ELBOW_X = int(LEFT_ELBOW.x * image_width)
  LEFT_ELBOW_Y = int(LEFT_ELBOW.y * image_height)
  if (LEFT_ELBOW.visibility < 0.5):
    detection = False

  RIGHT_WRIST = landmark_pose[16]
  RIGHT_WRIST_X = int(RIGHT_WRIST.x * image_width)
  RIGHT_WRIST_Y = int(RIGHT_WRIST.y * image_height)
  if (RIGHT_WRIST.visibility < 0.5):
    detection = False

  LEFT_WRIST = landmark_pose[15]
  LEFT_WRIST_X = int(LEFT_WRIST.x * image_width)
  LEFT_WRIST_Y = int(LEFT_WRIST.y * image_height)
  if (LEFT_WRIST.visibility < 0.5):
    detection = False
  
  RIGHT_HIP = landmark_pose[24]
  RIGHT_HIP_X = int(RIGHT_HIP.x * image_width)
  RIGHT_HIP_Y = int(RIGHT_HIP.y * image_height)
  if (RIGHT_HIP.visibility < 0.5):
    detection = False
  
  LEFT_HIP = landmark_pose[23]
  LEFT_HIP_X = int(LEFT_HIP.x * image_width)
  LEFT_HIP_Y = int(LEFT_HIP.y * image_height)
  if (LEFT_HIP.visibility < 0.5):
    detection = False

  RIGHT_KNEE = landmark_pose[26]
  RIGHT_KNEE_X = int(RIGHT_KNEE.x * image_width)
  RIGHT_KNEE_Y = int(RIGHT_KNEE.y * image_height)
  if (RIGHT_KNEE.visibility < 0.5):
    detection = False
  
  LEFT_KNEE = landmark_pose[25]
  LEFT_KNEE_X = int(LEFT_KNEE.x * image_width)
  LEFT_KNEE_Y = int(LEFT_KNEE.y * image_height)
  if (LEFT_KNEE.visibility < 0.5):
    detection = False

  RIGHT_ANKLE = landmark_pose[28]
  RIGHT_ANKLE_X = int(RIGHT_ANKLE.x * image_width)
  RIGHT_ANKLE_Y = int(RIGHT_ANKLE.y * image_height)
  if (RIGHT_ANKLE.visibility < 0.5):
    detection = False
  
  LEFT_ANKLE = landmark_pose[27]
  LEFT_ANKLE_X = int(LEFT_ANKLE.x * image_width)
  LEFT_ANKLE_Y = int(LEFT_ANKLE.y * image_height)
  if (LEFT_ANKLE.visibility < 0.5):
    detection = False

  return detection

def StartEngine(cap):
  DURATION = 100
  with mp_pose.Pose(min_detection_confidence=0.8,min_tracking_confidence=0.5) as pose:
    frame_count = 0

    while cap.isOpened():
      ret, image = cap.read()

      image_copy = image.copy()
      if not ret:
        print("Ignoring empty camera frame.")
        continue
      image_copy.flags.writeable = False
      image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
      results = pose.process(image_copy)

      if results.pose_landmarks:
        landmark_pose = results.pose_landmarks.landmark
        image_copy.flags.writeable = True

        Isdetected = EvalDetect(image_copy,landmark_pose)

        if (Isdetected):
          frame_count += 1
        else:
          frame_count = 0

        image_copy.flags.writeable = True
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
          image_copy,
          results.pose_landmarks,
          mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        cv2.rectangle(image_copy, (0,0), (350, 60), (245, 117, 16), cv2.FILLED)

        # Display Probability
        # cv2.putText(image_copy, 'STATUS',(15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if (frame_count == 0):
          cv2.putText(image_copy, "Finding ALL joints..", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
          cv2.putText(image_copy, 'loading : {}%'.format(int((frame_count/DURATION)*100)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Wake up',image_copy)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
          exit(0)

        if (frame_count == DURATION):
          break
    cv2.destroyAllWindows()