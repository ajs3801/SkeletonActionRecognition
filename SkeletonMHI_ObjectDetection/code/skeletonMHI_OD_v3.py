# DetectActionv3 : squat lunge stand pushup lying
# [DataSet]
# Squat : 100 | Lunge : 100 | Pushup : 100 | Lying : 58 | Stand : 46
# [Label Infor]
# Lunge - 16
# Squat - 15
# Stand - 19
# Lying - 17
# Pushup - 18

from asyncio import FastChildWatcher
import torch
import numpy as np
import cv2
import mediapipe as mp
import math

from zmq import PUSH

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

SQUAT = 0
LUNGE = 1
PUSHUP = 2
NONE = 3
GOOD = 4
NORMAL = 5
BAD = 6

MHI_DURATION = 30

LIST_COORD_RIGHT_SHOULDER = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_LEFT_SHOULDER = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_CENTER_SHOULDER = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_RIGHT_ELBOW = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_LEFT_ELBOW = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_RIGHT_WRIST = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_LEFT_WRIST = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_RIGHT_HIP = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_LEFT_HIP = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_CENTER_HIP = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_RIGHT_KNEE =[(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_LEFT_KNEE = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_RIGHT_ANKLE = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]
LIST_COORD_LEFT_ANKLE = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)]


# 30개의 프레임  넣기
COLOR = []
RADIUS = []
for idx in range(1,MHI_DURATION+1):
  ratio = (idx-1)/(MHI_DURATION-1)
  result = 15*ratio
  result += 3
  RADIUS.append(int(result))
RADIUS.reverse()

MHI_DURATION_FIRST = 15
for result_idx in range(1,MHI_DURATION_FIRST+1):
    ratio = (result_idx-1)/(MHI_DURATION_FIRST-1)
    COLOR.append((0,int(255*ratio), int(255*(1-ratio))))

MHI_DURATION_SECOND = 15
for result_idx in range(1,MHI_DURATION_SECOND+1):
    ratio = (result_idx-1)/(MHI_DURATION_SECOND-1)
    COLOR.append((int(255*ratio), int(255*(1-ratio)), 0))

def findDistance(x1,y1,x2,y2):
  return math.sqrt(math.pow(x1-x2,2)+ math.pow(y1-y2,2))

# Calculate angle.
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

  RIGHT_WRIST = landmark_pose[16]
  RIGHT_WRIST_X = int(RIGHT_WRIST.x * image_width)
  RIGHT_WRIST_Y = int(RIGHT_WRIST.y * image_height)
  if (RIGHT_WRIST.visibility < 0.5):
    RIGHT_WRIST_X = 1
    RIGHT_WRIST_Y = 1

  LEFT_WRIST = landmark_pose[15]
  LEFT_WRIST_X = int(LEFT_WRIST.x * image_width)
  LEFT_WRIST_Y = int(LEFT_WRIST.y * image_height)
  if (LEFT_WRIST.visibility < 0.5):
    LEFT_WRIST_X = 1
    LEFT_WRIST_X = 1
  
  LEFT_ELBOW = landmark_pose[13]
  LEFT_ELBOW_X = int(LEFT_ELBOW.x * image_width)
  LEFT_ELBOW_Y = int(LEFT_ELBOW.y * image_height)
  if (LEFT_ELBOW.visibility < 0.5):
    LEFT_ELBOW_X = 1
    LEFT_ELBOW_Y = 1

  RIGHT_ELBOW = landmark_pose[14]
  RIGHT_ELBOW_X = int(RIGHT_ELBOW.x * image_width)
  RIGHT_ELBOW_Y = int(RIGHT_ELBOW.y * image_height)
  if (RIGHT_ELBOW.visibility < 0.5):
    RIGHT_ELBOW_X = 1
    RIGHT_ELBOW_Y = 1

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
  degreeOfLeftArm = int(findAngle(LEFT_WRIST_X,LEFT_WRIST_Y,LEFT_SHOULDER_X,LEFT_SHOULDER_Y,LEFT_ELBOW_X,LEFT_ELBOW_Y))
  degreeOfRightArm = int(findAngle(RIGHT_WRIST_X,RIGHT_WRIST_Y,RIGHT_SHOULDER_X,RIGHT_SHOULDER_Y,RIGHT_ELBOW_X,RIGHT_ELBOW_Y))

  resultOfSquat_left = 0
  resultOfSquat_right = 0

  if (degreeOfLeftLeg >= 120):
    resultOfSquat_left = BAD
  elif (degreeOfLeftLeg<120 and degreeOfLeftLeg>=50):
    resultOfSquat_left = NORMAL
  else:
    resultOfSquat_left = GOOD

  if (degreeOfRightLeg >= 120):
    resultOfSquat_right = BAD
  elif (degreeOfRightLeg<120 and degreeOfRightLeg>=50):
    resultOfSquat_right = NORMAL
  else:
    resultOfSquat_right = GOOD

  return resultOfSquat_left,resultOfSquat_right
def EvalulateStandPose(image,landmark_pose):
  image_height, image_width, _ = image.shape 

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

  if (degreeOfLeftLeg>=120 and degreeOfRightLeg>=120):
    return True
  else:
    return False

# list에 좌표 넣기
def InsertCoordinate(image,landmark_pose):
  image_height, image_width, _ = image.shape 
  cv2.rectangle(image, (0,0), (image_width,image_height), (0,0,0), cv2.FILLED)

  # 좌표를 얻어옴
  POPNUM = 29
  RIGHT_SHOULDER = landmark_pose[12]
  RIGHT_SHOULDER_X = int(RIGHT_SHOULDER.x * image_width)
  RIGHT_SHOULDER_Y = int(RIGHT_SHOULDER.y * image_height)
  if (RIGHT_SHOULDER.visibility < 0.5):
    RIGHT_SHOULDER_X = 0
    RIGHT_SHOULDER_Y = 0
  LIST_COORD_RIGHT_SHOULDER.insert(0,(RIGHT_SHOULDER_X,RIGHT_SHOULDER_Y))
  LIST_COORD_RIGHT_SHOULDER.pop(POPNUM)

  LEFT_SHOULDER = landmark_pose[11]
  LEFT_SHOULDER_X = int(LEFT_SHOULDER.x * image_width)
  LEFT_SHOULDER_Y = int(LEFT_SHOULDER.y * image_height)
  if (LEFT_SHOULDER.visibility < 0.5):
    LEFT_SHOULDER_X = 0
    LEFT_SHOULDER_Y = 0
  LIST_COORD_LEFT_SHOULDER.insert(0,(LEFT_SHOULDER_X,LEFT_SHOULDER_Y))
  LIST_COORD_LEFT_SHOULDER.pop(POPNUM)
  
  CENTER_SHOULDER_X = int((RIGHT_SHOULDER_X+LEFT_SHOULDER_X)/2)
  CENTER_SHOULDER_Y = int((RIGHT_SHOULDER_Y+LEFT_SHOULDER_Y)/2)
  LIST_COORD_CENTER_SHOULDER.insert(0,(CENTER_SHOULDER_X,CENTER_SHOULDER_Y))
  LIST_COORD_CENTER_SHOULDER.pop(POPNUM)  

  RIGHT_ELBOW = landmark_pose[14]
  RIGHT_ELBOW_X = int(RIGHT_ELBOW.x * image_width)
  RIGHT_ELBOW_Y = int(RIGHT_ELBOW.y * image_height)
  if (RIGHT_ELBOW.visibility < 0.5):
    RIGHT_ELBOW_X = 0
    RIGHT_ELBOW_Y = 0
  LIST_COORD_RIGHT_ELBOW.insert(0,(RIGHT_ELBOW_X,RIGHT_ELBOW_Y))
  LIST_COORD_RIGHT_ELBOW.pop(POPNUM)

  LEFT_ELBOW = landmark_pose[13]
  LEFT_ELBOW_X = int(LEFT_ELBOW.x * image_width)
  LEFT_ELBOW_Y = int(LEFT_ELBOW.y * image_height)
  if (LEFT_ELBOW.visibility < 0.5):
    LEFT_ELBOW_X = 0
    LEFT_ELBOW_Y = 0
  LIST_COORD_LEFT_ELBOW.insert(0,(LEFT_ELBOW_X,LEFT_ELBOW_Y))
  LIST_COORD_LEFT_ELBOW.pop(POPNUM)

  RIGHT_WRIST = landmark_pose[16]
  RIGHT_WRIST_X = int(RIGHT_WRIST.x * image_width)
  RIGHT_WRIST_Y = int(RIGHT_WRIST.y * image_height)
  if (RIGHT_WRIST.visibility < 0.5):
    RIGHT_WRIST_X = 0
    RIGHT_WRIST_Y = 0
  LIST_COORD_RIGHT_WRIST.insert(0,(RIGHT_WRIST_X,RIGHT_WRIST_Y))
  LIST_COORD_RIGHT_WRIST.pop(POPNUM)

  LEFT_WRIST = landmark_pose[15]
  LEFT_WRIST_X = int(LEFT_WRIST.x * image_width)
  LEFT_WRIST_Y = int(LEFT_WRIST.y * image_height)
  if (LEFT_WRIST.visibility < 0.5):
    LEFT_WRIST_X = 0
    LEFT_WRIST_X = 0
  LIST_COORD_LEFT_WRIST.insert(0,(LEFT_WRIST_X,LEFT_WRIST_Y))
  LIST_COORD_LEFT_WRIST.pop(POPNUM)
  
  RIGHT_HIP = landmark_pose[24]
  RIGHT_HIP_X = int(RIGHT_HIP.x * image_width)
  RIGHT_HIP_Y = int(RIGHT_HIP.y * image_height)
  if (RIGHT_HIP.visibility < 0.5):
    RIGHT_HIP_X = 0
    RIGHT_HIP_Y = 0
  LIST_COORD_RIGHT_HIP.insert(0,(RIGHT_HIP_X,RIGHT_HIP_Y))
  LIST_COORD_RIGHT_HIP.pop(POPNUM)
  
  LEFT_HIP = landmark_pose[23]
  LEFT_HIP_X = int(LEFT_HIP.x * image_width)
  LEFT_HIP_Y = int(LEFT_HIP.y * image_height)
  if (LEFT_HIP.visibility < 0.5):
    LEFT_HIP_X = 0
    LEFT_HIP_Y = 0
  LIST_COORD_LEFT_HIP.insert(0,(LEFT_HIP_X,LEFT_HIP_Y))
  LIST_COORD_LEFT_HIP.pop(POPNUM)

  CENTER_HIP_X = int((RIGHT_HIP_X+LEFT_HIP_X)/2)
  CENTER_HIP_Y = int((RIGHT_HIP_Y+LEFT_HIP_Y)/2) 
  LIST_COORD_CENTER_HIP.insert(0,(CENTER_HIP_X,CENTER_HIP_Y))
  LIST_COORD_CENTER_HIP.pop(POPNUM)

  RIGHT_KNEE = landmark_pose[26]
  RIGHT_KNEE_X = int(RIGHT_KNEE.x * image_width)
  RIGHT_KNEE_Y = int(RIGHT_KNEE.y * image_height)
  if (RIGHT_KNEE.visibility < 0.5):
    RIGHT_KNEE_X = 0
    RIGHT_KNEE_Y = 0
  LIST_COORD_RIGHT_KNEE.insert(0,(RIGHT_KNEE_X,RIGHT_KNEE_Y))
  LIST_COORD_RIGHT_KNEE.pop(POPNUM)
  
  LEFT_KNEE = landmark_pose[25]
  LEFT_KNEE_X = int(LEFT_KNEE.x * image_width)
  LEFT_KNEE_Y = int(LEFT_KNEE.y * image_height)
  if (LEFT_KNEE.visibility < 0.5):
    LEFT_KNEE_X = 0
    LEFT_KNEE_Y = 0
  LIST_COORD_LEFT_KNEE.insert(0,(LEFT_KNEE_X,LEFT_KNEE_Y))
  LIST_COORD_LEFT_KNEE.pop(POPNUM)

  RIGHT_ANKLE = landmark_pose[28]
  RIGHT_ANKLE_X = int(RIGHT_ANKLE.x * image_width)
  RIGHT_ANKLE_Y = int(RIGHT_ANKLE.y * image_height)
  if (RIGHT_ANKLE.visibility < 0.5):
    RIGHT_ANKLE_X = 0
    RIGHT_ANKLE_Y = 0
  LIST_COORD_RIGHT_ANKLE.insert(0,(RIGHT_ANKLE_X,RIGHT_ANKLE_Y))
  LIST_COORD_RIGHT_ANKLE.pop(POPNUM)
  
  LEFT_ANKLE = landmark_pose[27]
  LEFT_ANKLE_X = int(LEFT_ANKLE.x * image_width)
  LEFT_ANKLE_Y = int(LEFT_ANKLE.y * image_height)
  if (LEFT_ANKLE.visibility < 0.5):
    LEFT_ANKLE_X = 0
    LEFT_ANKLE_Y = 0 
  LIST_COORD_LEFT_ANKLE.insert(0,(LEFT_ANKLE_X,LEFT_ANKLE_Y))
  LIST_COORD_LEFT_ANKLE.pop(POPNUM)
  
  if (LIST_COORD_RIGHT_SHOULDER[0][0]!=0 and LIST_COORD_RIGHT_ELBOW[0][0]!=0):
    # 오른쪽 어깨 - 오른쪽 팔꿈치
    cv2.line(image, (LIST_COORD_RIGHT_SHOULDER[0][0] , LIST_COORD_RIGHT_SHOULDER[0][1]) , (LIST_COORD_RIGHT_ELBOW[0][0], LIST_COORD_RIGHT_ELBOW[0][1]), (0,0,255), 3, cv2.LINE_AA)
  
  if (LIST_COORD_RIGHT_WRIST[0][0]!=0 and LIST_COORD_RIGHT_ELBOW[0][0]!=0):
    # 오른쪽 손목 - 오른쪽 팔꿈치
    cv2.line(image, (LIST_COORD_RIGHT_WRIST[0][0] , LIST_COORD_RIGHT_WRIST[0][1]) , (LIST_COORD_RIGHT_ELBOW[0][0], LIST_COORD_RIGHT_ELBOW[0][1]), (0,0,255), 3, cv2.LINE_AA)
  
  if (LIST_COORD_RIGHT_SHOULDER[0][0]!=0 and LIST_COORD_CENTER_SHOULDER[0][0]!=0):
    # 오른쪽 어깨 - 중어깨
    cv2.line(image, (LIST_COORD_RIGHT_SHOULDER[0][0] , LIST_COORD_RIGHT_SHOULDER[0][1]) , (LIST_COORD_CENTER_SHOULDER[0][0], LIST_COORD_CENTER_SHOULDER[0][1]), (0,0,255), 3, cv2.LINE_AA)

  if (LIST_COORD_CENTER_SHOULDER[0][0]!=0 and LIST_COORD_LEFT_SHOULDER[0][0]!=0):
    # 중어깨 - 왼쪽 어깨
    cv2.line(image, (LIST_COORD_CENTER_SHOULDER[0][0] , LIST_COORD_CENTER_SHOULDER[0][1]) , (LIST_COORD_LEFT_SHOULDER[0][0], LIST_COORD_LEFT_SHOULDER[0][1]), (0,0,255), 3, cv2.LINE_AA)
  
  if (LIST_COORD_LEFT_SHOULDER[0][0]!=0 and LIST_COORD_LEFT_ELBOW[0][0]!=0):
    # 왼쪽 어깨 - 왼쪽 팔꿈치
    cv2.line(image, (LIST_COORD_LEFT_SHOULDER[0][0], LIST_COORD_LEFT_SHOULDER[0][1]) , (LIST_COORD_LEFT_ELBOW[0][0], LIST_COORD_LEFT_ELBOW[0][1]), (0,0,255), 3, cv2.LINE_AA)
  
  if (LIST_COORD_LEFT_ELBOW[0][0]!=0 and LIST_COORD_LEFT_WRIST[0][0]!=0):
    # 왼쪽 팔꿈치 - 왼쪽 손목
    cv2.line(image, (LIST_COORD_LEFT_ELBOW[0][0], LIST_COORD_LEFT_ELBOW[0][1]) , (LIST_COORD_LEFT_WRIST[0][0], LIST_COORD_LEFT_WRIST[0][1]), (0,0,255), 3, cv2.LINE_AA)
  if (LIST_COORD_CENTER_SHOULDER[0][0]!=0 and LIST_COORD_CENTER_HIP[0][0]!=0):  
    # 중어꺠 - 중덩이
    cv2.line(image, (LIST_COORD_CENTER_SHOULDER[0][0] , LIST_COORD_CENTER_SHOULDER[0][1]) , (LIST_COORD_CENTER_HIP[0][0], LIST_COORD_CENTER_HIP[0][1]), (0,0,255), 3, cv2.LINE_AA)
  if (LIST_COORD_RIGHT_HIP[0][0]!=0 and LIST_COORD_CENTER_HIP[0][0]!=0):
    # 오른쪽 엉덩이 - 중덩이
    cv2.line(image, (LIST_COORD_RIGHT_HIP[0][0] , LIST_COORD_RIGHT_HIP[0][1]) , (LIST_COORD_CENTER_HIP[0][0], LIST_COORD_CENTER_HIP[0][1]), (0,0,255), 3, cv2.LINE_AA)
  if (LIST_COORD_LEFT_HIP[0][0]!=0 and LIST_COORD_CENTER_HIP[0][0]!=0):
    # 왼쪽 엉덩이 - 중덩이
    cv2.line(image, (LIST_COORD_LEFT_HIP[0][0] , LIST_COORD_LEFT_HIP[0][1]) , (LIST_COORD_CENTER_HIP[0][0], LIST_COORD_CENTER_HIP[0][1]), (0,0,255), 3, cv2.LINE_AA)
  if (LIST_COORD_RIGHT_HIP[0][0]!=0 and LIST_COORD_RIGHT_KNEE[0][0]!=0):
    # 오른쪽 엉덩이 - 오른쪽 무릎
    cv2.line(image, (LIST_COORD_RIGHT_HIP[0][0] , LIST_COORD_RIGHT_HIP[0][1]) , (LIST_COORD_RIGHT_KNEE[0][0], LIST_COORD_RIGHT_KNEE[0][1]), (0,0,255), 3, cv2.LINE_AA)
  if (LIST_COORD_RIGHT_KNEE[0][0]!=0 and LIST_COORD_RIGHT_ANKLE[0][0]!=0):
    # 오른쪽 무릎 - 오른쪽 발목
    cv2.line(image, (LIST_COORD_RIGHT_KNEE[0][0] , LIST_COORD_RIGHT_KNEE[0][1]) , (LIST_COORD_RIGHT_ANKLE[0][0], LIST_COORD_RIGHT_ANKLE[0][1]), (0,0,255), 3, cv2.LINE_AA)
  if (LIST_COORD_LEFT_HIP[0][0]!=0 and LIST_COORD_LEFT_KNEE[0][0]!=0):
    # 왼쪽 엉덩이 - 왼쪽 무릎
    cv2.line(image, (LIST_COORD_LEFT_HIP[0][0], LIST_COORD_LEFT_HIP[0][1]) , (LIST_COORD_LEFT_KNEE[0][0], LIST_COORD_LEFT_KNEE[0][1]), (0,0,255), 3, cv2.LINE_AA)
  if (LIST_COORD_LEFT_KNEE[0][0]!=0 and LIST_COORD_LEFT_ANKLE[0][0]!=0):
    # 왼쪽 무릎 - 왼쪽 발목
    cv2.line(image, (LIST_COORD_LEFT_KNEE[0][0] , LIST_COORD_LEFT_KNEE[0][1]) , (LIST_COORD_LEFT_ANKLE[0][0], LIST_COORD_LEFT_ANKLE[0][1]), (0,0,255), 3, cv2.LINE_AA)

  for idx in range(MHI_DURATION):
    cv2.circle(image, (LIST_COORD_RIGHT_SHOULDER[idx][0],LIST_COORD_RIGHT_SHOULDER[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_RIGHT_ANKLE[idx][0],LIST_COORD_RIGHT_ANKLE[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_RIGHT_ELBOW[idx][0],LIST_COORD_RIGHT_ELBOW[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_RIGHT_HIP[idx][0],LIST_COORD_RIGHT_HIP[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_RIGHT_KNEE[idx][0],LIST_COORD_RIGHT_KNEE[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_RIGHT_WRIST[idx][0],LIST_COORD_RIGHT_WRIST[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_LEFT_SHOULDER[idx][0],LIST_COORD_LEFT_SHOULDER[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_LEFT_ANKLE[idx][0],LIST_COORD_LEFT_ANKLE[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_LEFT_ELBOW[idx][0],LIST_COORD_LEFT_ELBOW[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_LEFT_KNEE[idx][0],LIST_COORD_LEFT_KNEE[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_LEFT_WRIST[idx][0],LIST_COORD_LEFT_WRIST[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 
    cv2.circle(image, (LIST_COORD_LEFT_HIP[idx][0],LIST_COORD_LEFT_HIP[idx][1]), RADIUS[idx] , COLOR[idx] , cv2.FILLED , cv2.LINE_AA) 

  cv2.circle(image, (0,0), 18, (0,0,0),cv2.FILLED, cv2.LINE_AA)

  return image

def ActionPerformed(prev,cur):
  if (prev == "squat" and cur == "stand"):
    return SQUAT
  
  elif (prev == "lunge" and cur == "stand"):
    return LUNGE

  elif (prev == "pushup" and cur == "lying"):
    return PUSHUP

  else:
    return NONE

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
  with mp_pose.Pose(min_detection_confidence=0.8,min_tracking_confidence=0.5) as pose:
    frame_count = 0

    while cap.isOpened():
      ret, image = cap.read()
      cv2.imshow("Original",image)

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
        print(frame_count)

        image_copy.flags.writeable = True
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
          image_copy,
          results.pose_landmarks,
          mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        cv2.putText(image_copy, "{}/100".format(str(frame_count)) , (100,100), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
        cv2.imshow('Wake up',image_copy)
        cv2.waitKey(1)
        if (frame_count == 50):
          break
    cv2.destroyAllWindows()
def CropImages():
  X_Coord = np.array([])
  Y_Coord = np.array([])
  for i in range(MHI_DURATION):
    X_Coord = np.append(X_Coord,np.array(LIST_COORD_RIGHT_SHOULDER[i][0]))
    X_Coord = np.append(X_Coord,np.array(LIST_COORD_LEFT_SHOULDER[i][0]))
    X_Coord = np.append(X_Coord,np.array(LIST_COORD_CENTER_SHOULDER[i][0]))
    X_Coord = np.append(X_Coord,np.array(LIST_COORD_RIGHT_ELBOW[i][0]))
    X_Coord = np.append(X_Coord,np.array(LIST_COORD_LEFT_ELBOW[i][0]))
    X_Coord = np.append(X_Coord,np.array(LIST_COORD_RIGHT_WRIST[i][0]))
    X_Coord = np.append(X_Coord,np.array(LIST_COORD_LEFT_WRIST[i][0]))
    X_Coord = np.append(X_Coord,np.array(LIST_COORD_RIGHT_HIP[i][0]))
    X_Coord = np.append(X_Coord,np.array(LIST_COORD_LEFT_HIP[i][0]))
    X_Coord = np.append(X_Coord,np.array(LIST_COORD_CENTER_HIP[i][0]))
    X_Coord = np.append(X_Coord,np.array(LIST_COORD_RIGHT_KNEE[i][0]))
    X_Coord = np.append(X_Coord,np.array(LIST_COORD_LEFT_KNEE[i][0]))
    X_Coord = np.append(X_Coord,np.array(LIST_COORD_RIGHT_ANKLE[i][0]))
    X_Coord = np.append(X_Coord,np.array(LIST_COORD_LEFT_ANKLE[i][0]))

    Y_Coord = np.append(Y_Coord,np.array(LIST_COORD_RIGHT_SHOULDER[i][1]))
    Y_Coord = np.append(Y_Coord,np.array(LIST_COORD_LEFT_SHOULDER[i][1]))
    Y_Coord = np.append(Y_Coord,np.array(LIST_COORD_CENTER_SHOULDER[i][1]))
    Y_Coord = np.append(Y_Coord,np.array(LIST_COORD_RIGHT_ELBOW[i][1]))
    Y_Coord = np.append(Y_Coord,np.array(LIST_COORD_LEFT_ELBOW[i][1]))
    Y_Coord = np.append(Y_Coord,np.array(LIST_COORD_RIGHT_WRIST[i][1]))
    Y_Coord = np.append(Y_Coord,np.array(LIST_COORD_LEFT_WRIST[i][1]))
    Y_Coord = np.append(Y_Coord,np.array(LIST_COORD_RIGHT_HIP[i][1]))
    Y_Coord = np.append(Y_Coord,np.array(LIST_COORD_LEFT_HIP[i][1]))
    Y_Coord = np.append(Y_Coord,np.array(LIST_COORD_CENTER_HIP[i][1]))
    Y_Coord = np.append(Y_Coord,np.array(LIST_COORD_RIGHT_KNEE[i][1]))
    Y_Coord = np.append(Y_Coord,np.array(LIST_COORD_LEFT_KNEE[i][1]))
    Y_Coord = np.append(Y_Coord,np.array(LIST_COORD_RIGHT_ANKLE[i][1]))
    Y_Coord = np.append(Y_Coord,np.array(LIST_COORD_LEFT_ANKLE[i][1]))
  
  X_Max = np.max(X_Coord)
  X_Min = np.min(X_Coord)

  Y_Max = np.max(Y_Coord)
  Y_Min = np.min(Y_Coord)

  return X_Max,X_Min,Y_Max,Y_Min

model = torch.hub.load('ultralytics/yolov5', 'custom', path='./model/ActionDetectionBestv3.pt', force_reload=True)

NumSquat = 0
NumLunge = 0
NumPushup = 0

actionList = [("stand",5),("stand",5),("stand",5)]

cap = cv2.VideoCapture("TestVideo/202208021432_original_SPL.avi")
# cap = cv2.VideoCapture(0)
# wake up
# StartEngine(cap)
# detection
with mp_pose.Pose(min_detection_confidence=0.8,min_tracking_confidence=0.5) as pose:
  previous = "stand"
  Current = "stand"
  frame_count = 0
  while cap.isOpened():
      ret, frame = cap.read()
      frame_count += 1
      image_height, image_width, _ = frame.shape

      cv2.imshow('Original',frame)

      frame.flags.writeable = False
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      results = pose.process(frame)

      frame.flags.writeable = True
      if results.pose_landmarks:
        landmark_pose = results.pose_landmarks.landmark

        frame = InsertCoordinate(frame,landmark_pose)
        frame = cv2.resize(frame, None, fx=0.25 , fy=0.25) #0.5배로 축소
        results_yolo = model(frame)

        labels, cord = results_yolo.xyxyn[0][:, -1], results_yolo.xyxyn[0][:, :-1]
        n = len(labels)

        x_shape, y_shape = frame.shape[1], frame.shape[0]

        squat, stand, lunge, lying, pushup = 0,0,0,0,0
        SQUAT, STAND, LUNGE, LYING, PUSHUP = 15,19,16,17,18

        action = "None"
        dictInfor = {"squat":0, "stand":0, "lunge":0, "lying":0, "pushup":0}

        for i in range(n):
            row = cord[i]
            if row[4] >= 0.5:
                
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                # if (x2-x2 < 30 or y2-y1<30):
                #   continue
                # print('width : {} | height : {}'.format(x2-x1,y2-y1))
                
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)

                if (labels[i] == SQUAT):
                  dictInfor["squat"] += 1
                
                if (labels[i] == STAND):
                  dictInfor["stand"] += 1

                if (labels[i] == LUNGE):
                  dictInfor["lunge"] += 1

                if (labels[i] == LYING):
                  dictInfor["lying"] += 1

                if (labels[i] == PUSHUP):
                  dictInfor["pushup"] += 1

        action = max(dictInfor, key = dictInfor.get)
        
        # if (action != actionList[0][0]):
        #   actionList.pop(len(actionList)-1)
        #   actionList.insert(0,(str(action),int(0)))

        # else:
        #   getAction = str(actionList[0][0])
        #   getFrameCount = int(actionList[0][1])
        #   getFrameCount += 1
        #   actionList.pop(0)

        #   actionList.insert(0,(str(getAction),int(getFrameCount)))
        
        # print(actionList)
        # print(actionList[1][0])
        # print(actionList[2][0]

        # try:
        #   FrameCountIndex1 = int(actionList[1][1])
        #   if (FrameCountIndex1 < 2):
        #     actionList.pop(1)
        # except:
        #   pass

        # try:
        #   FrameCountIndex2 = int(actionList[2][1])
        #   if (FrameCountIndex2 < 2):
        #     actionList.pop(2)
        # except:
        #   pass
        Current = action
        # print(actionList)

        frame = cv2.resize(frame, None, fx=4 , fy=4) #다시 확대

        doAction = ActionPerformed(previous,Current)

        if (doAction == SQUAT):
          NumSquat += 1
        elif (doAction == LUNGE):
          NumLunge += 1
        elif (doAction == PUSHUP):
          NumPushup += 1
        
        squat_left,squat_right = EvalulateSquatPose(frame,landmark_pose)
        dictEval = {"6":"Bad", "5":"Normal", "4":"GOOD"}
        if (action == "squat"):
          cv2.putText(frame, "Action : {}".format(str(dictEval[str(squat_right)])+" "+ action) , (100,100), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
        else:
          cv2.putText(frame, "Action : {}".format(action) , (100,100), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
        cv2.putText(frame, "Squat: {}".format(str(NumSquat)) , (100,150), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
        cv2.putText(frame, "Lunge: {}".format(str(NumLunge)) , (100,200), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
        cv2.putText(frame, "Pushup: {}".format(str(NumPushup)) , (100,250), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
        previous = Current
        

      cv2.imshow('skeletonMHI', frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)