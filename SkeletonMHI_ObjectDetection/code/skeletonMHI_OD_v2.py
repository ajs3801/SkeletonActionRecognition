import torch
import numpy as np
import cv2
import mediapipe as mp
import math

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

# 30개의 프레임 색깔 넣기
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

def dlist(x1,y1,x2,y2):
  return math.sqrt(math.pow(x1-x2,2)+ math.pow(y1-y2,2))

# list에 좌표 넣기
def InsertCoordinate(image,landmark_pose):
  image_height, image_width, _ = image.shape 
  cv2.rectangle(image, (0,0), (image_width,image_height), (0,0,0), cv2.FILLED)

  # 좌표를 얻어옴
  RIGHT_SHOULDER = landmark_pose[12]
  RIGHT_SHOULDER_X = int(RIGHT_SHOULDER.x * image_width)
  RIGHT_SHOULDER_Y = int(RIGHT_SHOULDER.y * image_height)
  if (RIGHT_SHOULDER.visibility < 0.5):
    RIGHT_SHOULDER_X = 0
    RIGHT_SHOULDER_Y = 0
  LIST_COORD_RIGHT_SHOULDER.insert(0,(RIGHT_SHOULDER_X,RIGHT_SHOULDER_Y))
  LIST_COORD_RIGHT_SHOULDER.pop(29)

  LEFT_SHOULDER = landmark_pose[11]
  LEFT_SHOULDER_X = int(LEFT_SHOULDER.x * image_width)
  LEFT_SHOULDER_Y = int(LEFT_SHOULDER.y * image_height)
  if (LEFT_SHOULDER.visibility < 0.5):
    LEFT_SHOULDER_X = 0
    LEFT_SHOULDER_Y = 0
  LIST_COORD_LEFT_SHOULDER.insert(0,(LEFT_SHOULDER_X,LEFT_SHOULDER_Y))
  LIST_COORD_LEFT_SHOULDER.pop(29)
  
  CENTER_SHOULDER_X = int((RIGHT_SHOULDER_X+LEFT_SHOULDER_X)/2)
  CENTER_SHOULDER_Y = int((RIGHT_SHOULDER_Y+LEFT_SHOULDER_Y)/2)
  LIST_COORD_CENTER_SHOULDER.insert(0,(CENTER_SHOULDER_X,CENTER_SHOULDER_Y))
  LIST_COORD_CENTER_SHOULDER.pop(29)  

  RIGHT_ELBOW = landmark_pose[14]
  RIGHT_ELBOW_X = int(RIGHT_ELBOW.x * image_width)
  RIGHT_ELBOW_Y = int(RIGHT_ELBOW.y * image_height)
  if (RIGHT_ELBOW.visibility < 0.5):
    RIGHT_ELBOW_X = 0
    RIGHT_ELBOW_Y = 0
  LIST_COORD_RIGHT_ELBOW.insert(0,(RIGHT_ELBOW_X,RIGHT_ELBOW_Y))
  LIST_COORD_RIGHT_ELBOW.pop(29)

  LEFT_ELBOW = landmark_pose[13]
  LEFT_ELBOW_X = int(LEFT_ELBOW.x * image_width)
  LEFT_ELBOW_Y = int(LEFT_ELBOW.y * image_height)
  if (LEFT_ELBOW.visibility < 0.5):
    LEFT_ELBOW_X = 0
    LEFT_ELBOW_Y = 0
  LIST_COORD_LEFT_ELBOW.insert(0,(LEFT_ELBOW_X,LEFT_ELBOW_Y))
  LIST_COORD_LEFT_ELBOW.pop(29)

  RIGHT_WRIST = landmark_pose[16]
  RIGHT_WRIST_X = int(RIGHT_WRIST.x * image_width)
  RIGHT_WRIST_Y = int(RIGHT_WRIST.y * image_height)
  if (RIGHT_WRIST.visibility < 0.5):
    RIGHT_WRIST_X = 0
    RIGHT_WRIST_Y = 0
  LIST_COORD_RIGHT_WRIST.insert(0,(RIGHT_WRIST_X,RIGHT_WRIST_Y))
  LIST_COORD_RIGHT_WRIST.pop(29)

  LEFT_WRIST = landmark_pose[15]
  LEFT_WRIST_X = int(LEFT_WRIST.x * image_width)
  LEFT_WRIST_Y = int(LEFT_WRIST.y * image_height)
  if (LEFT_WRIST.visibility < 0.5):
    LEFT_WRIST_X = 0
    LEFT_WRIST_X = 0
  LIST_COORD_LEFT_WRIST.insert(0,(LEFT_WRIST_X,LEFT_WRIST_Y))
  LIST_COORD_LEFT_WRIST.pop(29)
  
  RIGHT_HIP = landmark_pose[24]
  RIGHT_HIP_X = int(RIGHT_HIP.x * image_width)
  RIGHT_HIP_Y = int(RIGHT_HIP.y * image_height)
  if (RIGHT_HIP.visibility < 0.5):
    RIGHT_HIP_X = 0
    RIGHT_HIP_Y = 0
  LIST_COORD_RIGHT_HIP.insert(0,(RIGHT_HIP_X,RIGHT_HIP_Y))
  LIST_COORD_RIGHT_HIP.pop(29)
  
  LEFT_HIP = landmark_pose[23]
  LEFT_HIP_X = int(LEFT_HIP.x * image_width)
  LEFT_HIP_Y = int(LEFT_HIP.y * image_height)
  if (LEFT_HIP.visibility < 0.5):
    LEFT_HIP_X = 0
    LEFT_HIP_Y = 0
  LIST_COORD_LEFT_HIP.insert(0,(LEFT_HIP_X,LEFT_HIP_Y))
  LIST_COORD_LEFT_HIP.pop(29)

  CENTER_HIP_X = int((RIGHT_HIP_X+LEFT_HIP_X)/2)
  CENTER_HIP_Y = int((RIGHT_HIP_Y+LEFT_HIP_Y)/2) 
  LIST_COORD_CENTER_HIP.insert(0,(CENTER_HIP_X,CENTER_HIP_Y))
  LIST_COORD_CENTER_HIP.pop(29)

  RIGHT_KNEE = landmark_pose[26]
  RIGHT_KNEE_X = int(RIGHT_KNEE.x * image_width)
  RIGHT_KNEE_Y = int(RIGHT_KNEE.y * image_height)
  if (RIGHT_KNEE.visibility < 0.5):
    RIGHT_KNEE_X = 0
    RIGHT_KNEE_Y = 0
  LIST_COORD_RIGHT_KNEE.insert(0,(RIGHT_KNEE_X,RIGHT_KNEE_Y))
  LIST_COORD_RIGHT_KNEE.pop(29)
  
  LEFT_KNEE = landmark_pose[25]
  LEFT_KNEE_X = int(LEFT_KNEE.x * image_width)
  LEFT_KNEE_Y = int(LEFT_KNEE.y * image_height)
  if (LEFT_KNEE.visibility < 0.5):
    LEFT_KNEE_X = 0
    LEFT_KNEE_Y = 0
  LIST_COORD_LEFT_KNEE.insert(0,(LEFT_KNEE_X,LEFT_KNEE_Y))
  LIST_COORD_LEFT_KNEE.pop(29)

  RIGHT_ANKLE = landmark_pose[28]
  RIGHT_ANKLE_X = int(RIGHT_ANKLE.x * image_width)
  RIGHT_ANKLE_Y = int(RIGHT_ANKLE.y * image_height)
  if (RIGHT_ANKLE.visibility < 0.5):
    RIGHT_ANKLE_X = 0
    RIGHT_ANKLE_Y = 0
  LIST_COORD_RIGHT_ANKLE.insert(0,(RIGHT_ANKLE_X,RIGHT_ANKLE_Y))
  LIST_COORD_RIGHT_ANKLE.pop(29)
  
  LEFT_ANKLE = landmark_pose[27]
  LEFT_ANKLE_X = int(LEFT_ANKLE.x * image_width)
  LEFT_ANKLE_Y = int(LEFT_ANKLE.y * image_height)
  if (LEFT_ANKLE.visibility < 0.5):
    LEFT_ANKLE_X = 0
    LEFT_ANKLE_Y = 0 
  LIST_COORD_LEFT_ANKLE.insert(0,(LEFT_ANKLE_X,LEFT_ANKLE_Y))
  LIST_COORD_LEFT_ANKLE.pop(29)

  
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

model = torch.hub.load('ultralytics/yolov5', 'custom', path='./model/ActionDetectionBestv2.pt', force_reload=True)

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.8,min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
      ret, frame = cap.read()

      image_height, image_width, _ = frame.shape

      frame.flags.writeable = False
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      results = pose.process(frame)

      frame.flags.writeable = True
      if results.pose_landmarks:
        landmark_pose = results.pose_landmarks.landmark

        frame = InsertCoordinate(frame,landmark_pose)

        results_yolo = model(frame)

        labels, cord = results_yolo.xyxyn[0][:, -1], results_yolo.xyxyn[0][:, :-1]
        n = len(labels)

        count = 0
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        squat = 0
        stand = 0
        lunge = 0
        SQUAT = 15
        STAND = 16
        LUNGE = 17
        action = "None"

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
                squat += 1
            
            if (labels[i] == STAND):
                stand += 1

            if (labels[i] == LUNGE):
                lunge += 1

            if (squat > stand and squat > lunge):
              action = 'squat'
            
            elif (lunge > stand and lunge > squat):
              action = 'lunge'

            else:
              action = 'stand'

        # print("squat : {} | stand : {} | count : {}".format(squat,stand,count))
        print("action : {}".format(action))
        cv2.putText(frame, "Action : {}".format(action) , (100,100), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
        # cv2.imshow('YOLO', np.squeeze(results.render()))
        cv2.imshow('skeletonMHI', frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)