import cv2

# image : openCV frame
# action : current action -> 문자열로 넣어주기
# NumSquat : 스쿼트 카운트
# NumLunge : 런지 카운트
# NumPushup : 푸쉬업 카운트
def DrawText(image,action,NumSquat,NumLunge,NumPushup):
  # Action Infor
  cv2.rectangle(image, (0,0), (240,300), (245, 117, 16), -1)
  cv2.putText(image,str(action), (80,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
  
  # Count Infor
  cv2.putText(image, "Squat  {}".format(str(NumSquat)) , (50,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
  cv2.putText(image, "Lunge  {}".format(str(NumLunge)) , (50,170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
  cv2.putText(image, "Pushup {}".format(str(NumPushup)) , (50,220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
  
  cv2.rectangle(image, (0,60), (240, 65), (255, 255, 255), -1)