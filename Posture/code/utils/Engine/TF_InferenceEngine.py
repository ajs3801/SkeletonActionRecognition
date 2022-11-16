import mediapipe as mp # Import mediapipe
import cv2 # Import opencv
import numpy as np
import csv
import pandas as pd
import warnings
import math
import sys
import os
import tensorflow as tf

from utils.Dictionary.initDict import initDict
#from ..EvaluatePose import EvaluateSquatPose as esp
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from EvaluatePose import EvaluateLungePose as elp
from EvaluatePose import EvaluatePushUpPose as epp
from EvaluatePose import EvaluateSquatPose as esp
from Const import const
import Dictionary

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

def TF_InferenceEngine(cap,MODEL):
  actions = np.array(['squat','lunge','pushup','stand','lying'])
  MODEL = tf.keras.models.load_model(MODEL)
  NumSquat,NumLunge,NumPushup = 0,0,0
  dict = Dictionary.initDict()

  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.5) as pose:
        prev = "stand"
        cur = "stand"

        while cap.isOpened():
            ret, frame = cap.read()
            
            if (ret == False):
              break

            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        
            
            # Make Detections
            results = pose.process(image)
            # print(results.face_landmarks)
            
            # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
            
            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(
              image,
              results.pose_landmarks,
              mp_pose.POSE_CONNECTIONS,
              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())


            extracted_data = []
            # Export coordinates
            # try:
            if results.pose_world_landmarks:
              # coordinate inference
              row = list(np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_world_landmarks.landmark]).flatten())
              # Append class name
              extracted_data.append(row)
              tf_predict = MODEL.predict(extracted_data)
              # print(actions[np.argmax(res)])
              # # Posture Recognition
              # if (body_language_class == "stand"):
              #   with mp_hands.Hands(model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
              #     pass
              cur = actions[np.argmax(tf_predict)]
              
              # # Posture Recognition
              # if (body_language_class == "stand"):
              #   with mp_hands.Hands(model_complexity=0,min_detection_confidence=0.5,min_tracking_confidence=0.5) as hands:
              #     pass

          
              
              # Workout assist program
              # if (cur == const.SQUAT_STRING):
              #   esp.EvalulateSquatPose(image,results.pose_landmarks.landmark)
              # elif (cur == const.LUNGE_STRING):
              #   pass
              # elif (cur == const.PUSHUP_STRING):
              #   pass
              
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

              # Get status box
              cv2.rectangle(image, (0,0), (250, 500), (245, 117, 16), -1)
              cv2.putText(image, cur
                          , (50,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

              # Display Probability
              # cv2.putText(image, 'prob'
              #             , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
              # cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
              #             , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

              cv2.putText(image, "Squat  {}".format(str(NumSquat)) , (50,100), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
              cv2.putText(image, "Lunge  {}".format(str(NumLunge)) , (50,150), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
              cv2.putText(image, "Pushup {}".format(str(NumPushup)) , (50,200), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)

            cv2.imshow('Skeleton Action Classifier', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()