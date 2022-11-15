# 90도 돌리면 스켈레톤 좌표 다르게 뽑히는지 실험

import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

extract = np.empty((1, 88))

video1 = '/Users/jaejoon/LGuplus/main_project/lstm/videos/lunge-down/05-202207141513-0-20-0.avi'
video2 = '/Users/jaejoon/LGuplus/main_project/lstm/videos_rotate90/lunge-down/05-202207141513-0-20-0.avi'
video3 = '/Users/jaejoon/LGuplus/main_project/lstm/videos_rotate270/lunge-down/05-202207141513-0-20-0.avi'

# 비디오 캡쳐 시작
cap = cv2.VideoCapture(video3)
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # 미디어파이프를 이용하여 스켈레톤 추출
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        temp = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_world_landmarks.landmark]).flatten(
        ) if results.pose_world_landmarks else np.zeros(132)

        # 얼굴을 제외한 22개의 랜드마크만 사용하기 위해 0~43번 인덱스 내용은 버림
        extract = np.append(extract, [temp[44:]], axis=0)

# 첫번째 열은 아무 의미 없는 값이 들어가있기 때문에 지워줌
extract = np.delete(extract, (0), axis=0)
extract = extract.astype(np.float32)

# 30 프레임에서 추출한 관절 정보들을 하나의 csv 파일로 저장
# 소수 다섯번째 자리까지만 저장
np.savetxt('./video3.csv', extract, delimiter=",", fmt='%.5f')
cap.release()
