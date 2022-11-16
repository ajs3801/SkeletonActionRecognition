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

dictEval = {"5":"Bad", "6":"Normal", "7":"Good"}

def findAngle(x1, y1, x2, y2, cx, cy):
  try:
    theta = math.atan((y2-cy)/(x2-cx))-math.atan((y1-cy)/(x1-cx))
    degree = int(180/math.pi)*abs(theta)
    return degree
  except:
    return 0

def EvalulateSquatPose(image,landmark_pose):
  image_height, image_width, _ = image.shape 

  # 좌표를 얻어옴
  RIGHT_SHOULDER = landmark_pose[12]
  RIGHT_SHOULDER_X = int(RIGHT_SHOULDER.x * image_width)
  RIGHT_SHOULDER_Y = int(RIGHT_SHOULDER.y * image_height)
  if (RIGHT_SHOULDER.visibility < 0.5):
    RIGHT_SHOULDER_X = 1
    RIGHT_SHOULDER_Y = 1

  LEFT_SHOULDER = landmark_pose[11]
  LEFT_SHOULDER_X = int(LEFT_SHOULDER.x * image_width)
  LEFT_SHOULDER_Y = int(LEFT_SHOULDER.y * image_height)
  if (LEFT_SHOULDER.visibility < 0.5):
    LEFT_SHOULDER_X = 1
    LEFT_SHOULDER_Y = 1
  
  CENTER_SHOULDER_X = int((RIGHT_SHOULDER_X+LEFT_SHOULDER_X)/2)
  CENTER_SHOULDER_Y = int((RIGHT_SHOULDER_Y+LEFT_SHOULDER_Y)/2)

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
  
  CENTER_HIP_X = int((RIGHT_HIP_X+LEFT_HIP_X)/2)
  CENTER_HIP_Y = int((RIGHT_HIP_Y+LEFT_HIP_Y)/2)

  CENTER_X = int((CENTER_SHOULDER_X+CENTER_HIP_X)/2)
  CENTER_Y = int((CENTER_SHOULDER_Y+CENTER_HIP_Y)/2)

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
  
  ROW_HIP_X = CENTER_HIP_X
  ROW_HIP_Y = LEFT_KNEE

  degreeOfLeftLeg = 0
  degreeOfRightLeg = 0
  degreeOfLeftWaist = 0
  degreeOfRightWaist = 0

  try:
    degreeOfLeftLeg= int(findAngle(LEFT_ANKLE_X,LEFT_ANKLE_Y,LEFT_HIP_X,LEFT_HIP_Y,LEFT_KNEE_X,LEFT_KNEE_Y))
    degreeOfRightLeg = int(findAngle(RIGHT_ANKLE_X,RIGHT_ANKLE_Y,RIGHT_HIP_X,RIGHT_HIP_Y,RIGHT_KNEE_X,RIGHT_KNEE_Y))
    degreeOfLeftWaist = int(findAngle(LEFT_SHOULDER_X,LEFT_SHOULDER_Y,LEFT_HIP_X,LEFT_HIP_Y-5,LEFT_HIP_X,LEFT_HIP_Y))
    degreeOfRightWaist = int(findAngle(RIGHT_SHOULDER_X,RIGHT_SHOULDER_Y,RIGHT_HIP_X,RIGHT_HIP_Y-5,RIGHT_HIP_X,RIGHT_HIP_Y))
  except:
    pass
  # left = 0
  # right = 0
  try:
    if (degreeOfLeftWaist):
      if (degreeOfLeftWaist>=175):
        cv2.line(image, (CENTER_SHOULDER_X , CENTER_SHOULDER_Y) , (CENTER_HIP_X, CENTER_HIP_Y), (0,0,255), 3, cv2.LINE_AA)
        cv2.putText(image, "Waist down! {}".format(degreeOfLeftWaist) , (CENTER_X,CENTER_Y), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)

      elif (degreeOfLeftWaist<175 and degreeOfLeftWaist>=100):
        cv2.line(image, (CENTER_SHOULDER_X , CENTER_SHOULDER_Y) , (CENTER_HIP_X, CENTER_HIP_Y), (0,255,0), 3, cv2.LINE_AA)
        cv2.putText(image, "Waist GOOD {}".format(degreeOfLeftWaist) , (CENTER_X,CENTER_Y), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
      elif (degreeOfLeftWaist<100):
        cv2.line(image, (CENTER_SHOULDER_X , CENTER_SHOULDER_Y) , (CENTER_HIP_X, CENTER_HIP_Y), (0,0,255), 3, cv2.LINE_AA)
        cv2.putText(image, "Waist up! {}".format(degreeOfLeftWaist) , (CENTER_X,CENTER_Y), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
    
    if (degreeOfRightWaist):
      if (degreeOfRightWaist>=175):
        cv2.line(image, (CENTER_SHOULDER_X , CENTER_SHOULDER_Y) , (CENTER_HIP_X, CENTER_HIP_Y), (0,0,255), 3, cv2.LINE_AA)
        cv2.putText(image, "Waist down! {}".format(degreeOfRightWaist) , (CENTER_X,CENTER_Y), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)

      elif (degreeOfRightWaist<175 and degreeOfRightWaist>=100):
        cv2.line(image, (CENTER_SHOULDER_X , CENTER_SHOULDER_Y) , (CENTER_HIP_X, CENTER_HIP_Y), (0,255,0), 3, cv2.LINE_AA)
        cv2.putText(image, "Waist GOOD {}".format(degreeOfRightWaist) , (CENTER_X,CENTER_Y), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
      elif (degreeOfRightWaist<100):
        cv2.line(image, (CENTER_SHOULDER_X , CENTER_SHOULDER_Y) , (CENTER_HIP_X, CENTER_HIP_Y), (0,0,255), 3, cv2.LINE_AA)
        cv2.putText(image, "Waist up! {}".format(degreeOfRightWaist) , (CENTER_X,CENTER_Y), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)

    if (degreeOfLeftLeg):
      if (degreeOfLeftLeg >= 120):
        cv2.putText(image, "Bad" , (LEFT_KNEE_X,LEFT_KNEE_Y), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
      elif (degreeOfLeftLeg<120 and degreeOfLeftLeg>=80):
        cv2.putText(image, "Normal" , (LEFT_KNEE_X,LEFT_KNEE_Y), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,255), 3)
      else:
        cv2.putText(image, "Good" , (LEFT_KNEE_X,LEFT_KNEE_Y), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

    if (degreeOfRightLeg):
      if (degreeOfRightLeg >= 120):
        cv2.putText(image, "Bad" , (RIGHT_KNEE_X-5,RIGHT_KNEE_Y), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3)
      elif (degreeOfRightLeg<120 and degreeOfRightLeg>=80):
        cv2.putText(image, "Normal" , (RIGHT_KNEE_X-5,RIGHT_KNEE_Y), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,255), 3)
      else:
        cv2.putText(image, "Good" , (RIGHT_KNEE_X-5,RIGHT_KNEE_Y), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
  except:
    pass
  # cv2.putText(image, "{}".format(degreeOfLeftLeg) , (LEFT_KNEE_X-5,LEFT_KNEE_Y), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
  # cv2.putText(image, "{}".format(degreeOfRightLeg) , (RIGHT_KNEE_X-5,RIGHT_KNEE_Y), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

  # if (left < right):
  #   cv2.putText(image, "leg -> {} squat".format(str(dictEval[str(left)])) , (50,300), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
  # else:
  #   cv2.putText(image, "leg -> {} squat".format(str(dictEval[str(right)])) , (50,300), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)