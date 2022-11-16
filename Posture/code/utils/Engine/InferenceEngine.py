import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import numpy as np
import csv
import pickle 
import pandas as pd
import warnings
import math
import sys
import os
import timeit

from utils.Dictionary.initDict import initDict
import utils.Draw as Draw
#from ..EvaluatePose import EvaluateSquatPose as esp
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from EvaluatePose import EvaluateLungePose as elp
from EvaluatePose import EvaluatePushUpPose as epp
from EvaluatePose import EvaluateSquatPose as esp
from Const import const
import Dictionary
import MSE

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def IncreaseNum(increaseNum):
  increaseNum += 1
  return increaseNum

def ActionPerformed(prev,cur):
  if (prev == const.SQUAT_STRING and cur == const.STAND_STRING):
    return const.SQUAT_STRING
  
  elif (prev == const.LUNGE_STRING and cur == const.STAND_STRING):
    return const.LUNGE_STRING

  elif (prev == const.PUSHUP_STRING and cur == const.LYINGE_STRING):
    return const.PUSHUP_STRING

  else:
    return const.NONE_STRING

def InferenceEngine(cap,MODEL):
  NumSquat,NumLunge,NumPushup = 0,0,0
  dict = Dictionary.initDict()
  
  with open(MODEL, 'rb') as f:
    model = pickle.load(f)
  
  # Initiate holistic model
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.5) as pose:
        prev = "stand"
        cur = "stand"

        while cap.isOpened():
            ret, frame = cap.read()
            
            if (ret == False):
              break
            start_t = timeit.default_timer()
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        
            height, width, shape  = image.shape
            # Make Detections
            results = pose.process(image)
            # print(results.face_landmarks)
            
            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
            
            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # try:
            results = pose.process(image)
            image = Draw.DrawSkeleton(image,results.pose_landmarks.landmark,(203, 192, 255))
            # except:
            #   pass
            # Export coordinates
            # try:
            if results.pose_world_landmarks:
              # coordinate inference
              row = list(np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_world_landmarks.landmark]).flatten())
              X = pd.DataFrame([row])
              body_language_class = model.predict(X)[0]
              body_language_prob = model.predict_proba(X)[0]
              terminate_t = timeit.default_timer()
              FPS_realtime = int(1./(terminate_t - start_t ))
              # # Posture Recognition
              # if (body_language_class == "stand"):
              #   with mp_hands.Hands(model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
              #     pass

              cur = body_language_class
              
              # Eval count
              doAction = ""
              if (cur == const.STAND_STRING or cur == const.LYINGE_STRING):
                doAction = Dictionary.EvaluateDictAction(dict,cur)
                dict = Dictionary.initDict()
              else:
                Dictionary.IncreaseDict(dict,cur)
              
              if (doAction == const.SQUAT_STRING):
                NumSquat = IncreaseNum(NumSquat)

              elif (doAction == const.LUNGE_STRING):
                NumLunge = IncreaseNum(NumLunge)

              elif (doAction == const.PUSHUP_STRING):
                NumPushup = IncreaseNum(NumPushup)

              Draw.DrawText(image,cur,NumSquat,NumLunge,NumPushup)
              # Workout assist program
              if (cur == const.SQUAT_STRING):
                esp.EvalulateSquatPose(image,results.pose_landmarks.landmark)
                MSE.SquatMSE(image,row)
              elif (cur == const.LUNGE_STRING):
                elp.EvalulateLungePose(image,results.pose_landmarks.landmark)
                MSE.LungeMSE(image,row)
              elif (cur == const.PUSHUP_STRING):
                epp.EvalulatePushUpPose(image,results.pose_landmarks.landmark)
                MSE.PushupMSE(image,row)
              # # Get status box
              # cv2.rectangle(image, (0,0), (240,300), (245, 117, 16), -1)
              # cv2.putText(image, body_language_class.split(' ')[0]
              #             , (80,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
              
              # # Display Probability
              # # cv2.putText(image, 'prob'
              # #             , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
              # # cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
              # #             , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

              # cv2.putText(image, "Squat  {}".format(str(NumSquat)) , (50,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
              # cv2.putText(image, "Lunge  {}".format(str(NumLunge)) , (50,170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
              # cv2.putText(image, "Pushup {}".format(str(NumPushup)) , (50,220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

              # if (cur == const.SQUAT_STRING):
              #   MSE.SquatMSE(image,row)
              # elif (cur == const.LUNGE_STRING):
              #   MSE.LungeMSE(image,row)
              # elif (cur == const.PUSHUP_STRING):
              #   MSE.PushupMSE(image,row)
              
              # cv2.rectangle(image, (0,60), (240, 65), (255, 255, 255), -1)
              # remain = height - 300
              # remain = int(remain/4)
              # start_y = 275 - remain
              # for i in range(4):
              #   start_y += remain
              #   cv2.rectangle(image, (0,start_y), (240, start_y+5), (255, 255, 255), -1)
              # cv2.rectangle(image, (0,280), (240, 315), (255, 255, 255), -1)
              # cv2.rectangle(image, (0,345), (240, 350), (255, 255, 255), -1)
              # cv2.rectangle(image, (0,380), (240, 385), (255, 255, 255), -1)

              # cv2.putText(image, "fps {}".format(str(FPS_realtime)),(50,250), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
            cv2.imshow('Health Assist', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()