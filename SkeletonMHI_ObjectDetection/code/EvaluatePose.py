# DetectActionv3 : squat lunge stand pushup lying
# [DataSet]
# Squat : 100 | Lunge : 100 | Pushup : 100 | Lying : 58 | Stand : 46
# [Label Infor]
# Lunge - 16
# Squat - 15
# Stand - 19
# Lying - 17
# Pushup - 18

import torch
import numpy as np
import cv2
import mediapipe as mp
import math

from zmq import PUSH

SQUAT = 0
LUNGE = 1
PUSHUP = 2
NONE = 3
GOOD = 4
NORMAL = 5
BAD = 6

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

def EvalulatePose(image,landmark_pose):
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

  print(degreeOfLeftLeg,degreeOfRightLeg)
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

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

NumSquat = 0
NumLunge = 0
NumPushup = 0

cap = cv2.VideoCapture("./video/202207201402_original.avi")
with mp_pose.Pose(min_detection_confidence=0.8,min_tracking_confidence=0.5) as pose:
  previous = "stand"
  Current = "stand"
  while cap.isOpened():
      ret, frame = cap.read()

      image_height, image_width, _ = frame.shape

      frame.flags.writeable = False
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      results = pose.process(frame)

      frame.flags.writeable = True
      if results.pose_landmarks:
        landmark_pose = results.pose_landmarks.landmark

        action = max(dictInfor, key = dictInfor.get)
        Current = action
        # print("squat : {} | stand : {} | count : {}".format(squat,stand,count))
        # print("action : {}".format(action))
        # cv2.imshow('YOLO', np.squeeze(results.render()))
        frame = cv2.resize(frame, None, fx=2 , fy=2) #다시 확대
        cv2.putText(frame, "Action : {}".format(action) , (100,100), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

        doAction = ActionPerformed(previous,Current)

        if (doAction == SQUAT):
          NumSquat += 1
        elif (doAction == LUNGE):
          NumLunge += 1
        elif (doAction == PUSHUP):
          NumPushup += 1
        
        squat_left,squat_right = EvalulatePose(frame,landmark_pose)
        dictEval = {"6":"Bad", "5":"Normal", "4":"GOOD"}
        print("squat left : {}, squat right : {}".format(dictEval[str(squat_left)],dictEval[str(squat_right)]))
        cv2.putText(frame, "Squat: {}".format(str(NumSquat)) , (100,150), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
        cv2.putText(frame, "Lunge: {}".format(str(NumLunge)) , (100,200), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
        cv2.putText(frame, "Pushup: {}".format(str(NumPushup)) , (100,250), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)

        cv2.imshow('skeletonMHI', frame)
        previous = Current

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)