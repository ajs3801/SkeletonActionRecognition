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

def findAngle(x1, y1, x2, y2, cx, cy):
  division_degree_first = x2-cx
  if (division_degree_first <= 0):
    division_degree_first= 1

  division_degree_second = x1-cx
  if (division_degree_second <= 0):
    division_degree_second= 1

  theta = math.atan((y2-cy)/division_degree_first)-math.atan((y1-cy)/division_degree_second)
  degree = int(180/math.pi)*abs(theta)

  return degree

def EvalulateSquatPose(image,landmark_pose):
  image_height, image_width, _ = image.shape 

  # # 좌표를 얻어옴
  # RIGHT_SHOULDER = landmark_pose[12]
  # RIGHT_SHOULDER_X = int(RIGHT_SHOULDER.x * image_width)
  # RIGHT_SHOULDER_Y = int(RIGHT_SHOULDER.y * image_height)
  # if (RIGHT_SHOULDER.visibility < 0.5):
  #   RIGHT_SHOULDER_X = 1
  #   RIGHT_SHOULDER_Y = 1

  # LEFT_SHOULDER = landmark_pose[11]
  # LEFT_SHOULDER_X = int(LEFT_SHOULDER.x * image_width)
  # LEFT_SHOULDER_Y = int(LEFT_SHOULDER.y * image_height)
  # if (LEFT_SHOULDER.visibility < 0.5):
  #   LEFT_SHOULDER_X = 1
  #   LEFT_SHOULDER_Y = 1

  # RIGHT_WRIST = landmark_pose[16]
  # RIGHT_WRIST_X = int(RIGHT_WRIST.x * image_width)
  # RIGHT_WRIST_Y = int(RIGHT_WRIST.y * image_height)
  # if (RIGHT_WRIST.visibility < 0.5):
  #   RIGHT_WRIST_X = 1
  #   RIGHT_WRIST_Y = 1

  # LEFT_WRIST = landmark_pose[15]
  # LEFT_WRIST_X = int(LEFT_WRIST.x * image_width)
  # LEFT_WRIST_Y = int(LEFT_WRIST.y * image_height)
  # if (LEFT_WRIST.visibility < 0.5):
  #   LEFT_WRIST_X = 1
  #   LEFT_WRIST_X = 1
  
  # LEFT_ELBOW = landmark_pose[13]
  # LEFT_ELBOW_X = int(LEFT_ELBOW.x * image_width)
  # LEFT_ELBOW_Y = int(LEFT_ELBOW.y * image_height)
  # if (LEFT_ELBOW.visibility < 0.5):
  #   LEFT_ELBOW_X = 1
  #   LEFT_ELBOW_Y = 1

  # RIGHT_ELBOW = landmark_pose[14]
  # RIGHT_ELBOW_X = int(RIGHT_ELBOW.x * image_width)
  # RIGHT_ELBOW_Y = int(RIGHT_ELBOW.y * image_height)
  # if (RIGHT_ELBOW.visibility < 0.5):
  #   RIGHT_ELBOW_X = 1
  #   RIGHT_ELBOW_Y = 1

  RIGHT_HIP = landmark_pose[24]
  RIGHT_HIP_X = int(RIGHT_HIP.x * image_width)
  RIGHT_HIP_Y = int(RIGHT_HIP.y * image_height)
  if (RIGHT_HIP.visibility < 0.5):
    RIGHT_HIP_X = 1
    RIGHT_HIP_Y = 1
  
  LEFT_HIP = landmark_pose[23]
  LEFT_HIP_X = int(LEFT_HIP.x * image_width)
  LEFT_HIP_Y = int(LEFT_HIP.y * image_height)
  if (LEFT_HIP.visibility < 0.5):
    LEFT_HIP_X = 1
    LEFT_HIP_Y = 1

  RIGHT_ANKLE = landmark_pose[28]
  RIGHT_ANKLE_X = int(RIGHT_ANKLE.x * image_width)
  RIGHT_ANKLE_Y = int(RIGHT_ANKLE.y * image_height)
  if (RIGHT_ANKLE.visibility < 0.5):
    RIGHT_ANKLE_X = 1
    RIGHT_ANKLE_Y = 1
  
  LEFT_ANKLE = landmark_pose[27]
  LEFT_ANKLE_X = int(LEFT_ANKLE.x * image_width)
  LEFT_ANKLE_Y = int(LEFT_ANKLE.y * image_height)
  if (LEFT_ANKLE.visibility < 0.5):
    LEFT_ANKLE_X = 1
    LEFT_ANKLE_Y = 1

  RIGHT_KNEE = landmark_pose[26]
  RIGHT_KNEE_X = int(RIGHT_KNEE.x * image_width)
  RIGHT_KNEE_Y = int(RIGHT_KNEE.y * image_height)
  if (RIGHT_KNEE.visibility < 0.5):
    RIGHT_KNEE_X = 1
    RIGHT_KNEE_Y = 1
  
  LEFT_KNEE = landmark_pose[25]
  LEFT_KNEE_X = int(LEFT_KNEE.x * image_width)
  LEFT_KNEE_Y = int(LEFT_KNEE.y * image_height)
  if (LEFT_KNEE.visibility < 0.5):
    LEFT_KNEE_X = 1
    LEFT_KNEE_Y = 1
  
  degreeOfLeftLeg= int(findAngle(LEFT_ANKLE_X,LEFT_ANKLE_Y,LEFT_HIP_X,LEFT_HIP_Y,LEFT_KNEE_X,LEFT_KNEE_Y))
  degreeOfRightLeg = int(findAngle(RIGHT_ANKLE_X,RIGHT_ANKLE_Y,RIGHT_HIP_X,RIGHT_HIP_Y,RIGHT_KNEE_X,RIGHT_KNEE_Y))
  # degreeOfLeftArm = int(findAngle(LEFT_WRIST_X,LEFT_WRIST_Y,LEFT_SHOULDER_X,LEFT_SHOULDER_Y,LEFT_ELBOW_X,LEFT_ELBOW_Y))
  # degreeOfRightArm = int(findAngle(RIGHT_WRIST_X,RIGHT_WRIST_Y,RIGHT_SHOULDER_X,RIGHT_SHOULDER_Y,RIGHT_ELBOW_X,RIGHT_ELBOW_Y))

  resultOfSquat_left = 0
  resultOfSquat_right = 0

  # if (degreeOfLeftLeg >= 130):
  #   resultOfSquat_left = BAD
  # elif (degreeOfLeftLeg<130 and degreeOfLeftLeg>=90):
  #   resultOfSquat_left = NORMAL
  # else:
  #   resultOfSquat_left = GOOD

  # if (degreeOfRightLeg >= 130):
  #   resultOfSquat_right = BAD
  # elif (degreeOfRightLeg<130 and degreeOfRightLeg>=90):
  #   resultOfSquat_right = NORMAL
  # else:
  #   resultOfSquat_right = GOOD

  return resultOfSquat_left,resultOfSquat_right