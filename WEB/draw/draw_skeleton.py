import numpy as np
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# image : openCV frame

# landmark_pose : results.pose_landmarks.landmark
# [results = pose.process(image)]

# COLOR : skeleton Color -> (ex) (20THCIKNESS,192,255)


def draw_skeleton(image, landmark_pose, COLOR):
    image_height, image_width, _ = image.shape

    COLOR = (52, 0, 165)
    THICKNESS = 5
    # COLOR = (0, 0, 0)
    BLACK_COLOR = (255, 255, 255)
    RADIUS = 15
    THCIKNESS = 10
    RIGHT_SHOULDER = landmark_pose[12]
    RIGHT_SHOULDER_X = int(RIGHT_SHOULDER.x * image_width)
    RIGHT_SHOULDER_Y = int(RIGHT_SHOULDER.y * image_height)
    if (RIGHT_SHOULDER.visibility < 0.5):
        RIGHT_SHOULDER_X = 0
        RIGHT_SHOULDER_Y = 0
    else:
        cv2.circle(image, (RIGHT_SHOULDER_X, RIGHT_SHOULDER_Y),
                   RADIUS, BLACK_COLOR, cv2.FILLED, cv2.LINE_AA)  # 속이 꽉 찬 원
    LEFT_SHOULDER = landmark_pose[11]
    LEFT_SHOULDER_X = int(LEFT_SHOULDER.x * image_width)
    LEFT_SHOULDER_Y = int(LEFT_SHOULDER.y * image_height)
    if (LEFT_SHOULDER.visibility < 0.5):
        LEFT_SHOULDER_X = 0
        LEFT_SHOULDER_Y = 0
    else:
        cv2.circle(image, (LEFT_SHOULDER_X, LEFT_SHOULDER_Y),
                   RADIUS, BLACK_COLOR, cv2.FILLED, cv2.LINE_AA)  # 속이 꽉 찬 원
    CENTER_SHOULDER_X = int((RIGHT_SHOULDER_X+LEFT_SHOULDER_X)/2)
    CENTER_SHOULDER_Y = int((RIGHT_SHOULDER_Y+LEFT_SHOULDER_Y)/2)

    RIGHT_ELBOW = landmark_pose[14]
    RIGHT_ELBOW_X = int(RIGHT_ELBOW.x * image_width)
    RIGHT_ELBOW_Y = int(RIGHT_ELBOW.y * image_height)
    if (RIGHT_ELBOW.visibility < 0.5):
        RIGHT_ELBOW_X = 0
        RIGHT_ELBOW_Y = 0
    else:
        cv2.circle(image, (RIGHT_ELBOW_X, RIGHT_ELBOW_Y),
                   RADIUS, BLACK_COLOR, cv2.FILLED, cv2.LINE_AA)  # 속이 꽉 찬 원

    LEFT_ELBOW = landmark_pose[13]
    LEFT_ELBOW_X = int(LEFT_ELBOW.x * image_width)
    LEFT_ELBOW_Y = int(LEFT_ELBOW.y * image_height)
    if (LEFT_ELBOW.visibility < 0.5):
        LEFT_ELBOW_X = 0
        LEFT_ELBOW_Y = 0
    else:
        cv2.circle(image, (LEFT_ELBOW_X, LEFT_ELBOW_Y),
                   RADIUS, BLACK_COLOR, cv2.FILLED, cv2.LINE_AA)  # 속이 꽉 찬 원
    RIGHT_WRIST = landmark_pose[16]
    RIGHT_WRIST_X = int(RIGHT_WRIST.x * image_width)
    RIGHT_WRIST_Y = int(RIGHT_WRIST.y * image_height)
    if (RIGHT_WRIST.visibility < 0.5):
        RIGHT_WRIST_X = 0
        RIGHT_WRIST_Y = 0
    else:
        cv2.circle(image, (RIGHT_WRIST_X, RIGHT_WRIST_Y),
                   RADIUS, BLACK_COLOR, cv2.FILLED, cv2.LINE_AA)  # 속이 꽉 찬 원

    LEFT_WRIST = landmark_pose[15]
    LEFT_WRIST_X = int(LEFT_WRIST.x * image_width)
    LEFT_WRIST_Y = int(LEFT_WRIST.y * image_height)
    if (LEFT_WRIST.visibility < 0.5):
        LEFT_WRIST_X = 0
        LEFT_WRIST_X = 0
    else:
        cv2.circle(image, (LEFT_WRIST_X, LEFT_WRIST_Y),
                   RADIUS, BLACK_COLOR, cv2.FILLED, cv2.LINE_AA)  # 속이 꽉 찬 원

    RIGHT_HIP = landmark_pose[24]
    RIGHT_HIP_X = int(RIGHT_HIP.x * image_width)
    RIGHT_HIP_Y = int(RIGHT_HIP.y * image_height)
    if (RIGHT_HIP.visibility < 0.5):
        RIGHT_HIP_X = 0
        RIGHT_HIP_Y = 0
    else:
        cv2.circle(image, (RIGHT_HIP_X, RIGHT_HIP_Y),
                   RADIUS, BLACK_COLOR, cv2.FILLED, cv2.LINE_AA)  # 속이 꽉 찬 원

    LEFT_HIP = landmark_pose[23]
    LEFT_HIP_X = int(LEFT_HIP.x * image_width)
    LEFT_HIP_Y = int(LEFT_HIP.y * image_height)
    if (LEFT_HIP.visibility < 0.5):
        LEFT_HIP_X = 0
        LEFT_HIP_Y = 0
    else:
        cv2.circle(image, (LEFT_HIP_X, LEFT_HIP_Y),
                   RADIUS, BLACK_COLOR, cv2.FILLED, cv2.LINE_AA)  # 속이 꽉 찬 원

    CENTER_HIP_X = int((RIGHT_HIP_X+LEFT_HIP_X)/2)
    CENTER_HIP_Y = int((RIGHT_HIP_Y+LEFT_HIP_Y)/2)

    RIGHT_KNEE = landmark_pose[26]
    RIGHT_KNEE_X = int(RIGHT_KNEE.x * image_width)
    RIGHT_KNEE_Y = int(RIGHT_KNEE.y * image_height)
    if (RIGHT_KNEE.visibility < 0.5):
        RIGHT_KNEE_X = 0
        RIGHT_KNEE_Y = 0
    else:
        cv2.circle(image, (RIGHT_KNEE_X, RIGHT_KNEE_Y),
                   RADIUS, BLACK_COLOR, cv2.FILLED, cv2.LINE_AA)  # 속이 꽉 찬 원

    LEFT_KNEE = landmark_pose[25]
    LEFT_KNEE_X = int(LEFT_KNEE.x * image_width)
    LEFT_KNEE_Y = int(LEFT_KNEE.y * image_height)
    if (LEFT_KNEE.visibility < 0.5):
        LEFT_KNEE_X = 0
        LEFT_KNEE_Y = 0
    else:
        cv2.circle(image, (LEFT_KNEE_X, LEFT_KNEE_Y),
                   RADIUS, BLACK_COLOR, cv2.FILLED, cv2.LINE_AA)  # 속이 꽉 찬 원

    RIGHT_ANKLE = landmark_pose[28]
    RIGHT_ANKLE_X = int(RIGHT_ANKLE.x * image_width)
    RIGHT_ANKLE_Y = int(RIGHT_ANKLE.y * image_height)
    if (RIGHT_ANKLE.visibility < 0.5):
        RIGHT_ANKLE_X = 0
        RIGHT_ANKLE_Y = 0
    else:
        cv2.circle(image, (RIGHT_ANKLE_X, RIGHT_ANKLE_Y),
                   RADIUS, BLACK_COLOR, cv2.FILLED, cv2.LINE_AA)  # 속이 꽉 찬 원

    LEFT_ANKLE = landmark_pose[27]
    LEFT_ANKLE_X = int(LEFT_ANKLE.x * image_width)
    LEFT_ANKLE_Y = int(LEFT_ANKLE.y * image_height)
    if (LEFT_ANKLE.visibility < 0.5):
        LEFT_ANKLE_X = 0
        LEFT_ANKLE_Y = 0
    else:
        cv2.circle(image, (LEFT_ANKLE_X, LEFT_ANKLE_Y),
                   RADIUS, BLACK_COLOR, cv2.FILLED, cv2.LINE_AA)  # 속이 꽉 찬 원

    if (RIGHT_SHOULDER_X != 0 and RIGHT_ELBOW_X != 0):
        # 오른쪽 어깨 - 오른쪽 팔꿈치
        cv2.line(image, (RIGHT_SHOULDER_X, RIGHT_SHOULDER_Y),
                 (RIGHT_ELBOW_X, RIGHT_ELBOW_Y), COLOR, THCIKNESS, cv2.LINE_AA)

    if (RIGHT_WRIST_X != 0 and RIGHT_ELBOW_X != 0):
        # 오른쪽 손목 - 오른쪽 팔꿈치
        cv2.line(image, (RIGHT_WRIST_X, RIGHT_WRIST_Y),
                 (RIGHT_ELBOW_X, RIGHT_ELBOW_Y), COLOR, THCIKNESS, cv2.LINE_AA)

    if (RIGHT_SHOULDER_X != 0 and CENTER_SHOULDER_X != 0):
        # 오른쪽 어깨 - 중어깨
        cv2.line(image, (RIGHT_SHOULDER_X, RIGHT_SHOULDER_Y),
                 (CENTER_SHOULDER_X, CENTER_SHOULDER_Y), COLOR, THCIKNESS, cv2.LINE_AA)

    if (CENTER_SHOULDER_X != 0 and LEFT_SHOULDER_X != 0):
        # 중어깨 - 왼쪽 어깨
        cv2.line(image, (CENTER_SHOULDER_X, CENTER_SHOULDER_Y),
                 (LEFT_SHOULDER_X, LEFT_SHOULDER_Y), COLOR, THCIKNESS, cv2.LINE_AA)

    if (LEFT_SHOULDER_X != 0 and LEFT_ELBOW_X != 0):
        # 왼쪽 어깨 - 왼쪽 팔꿈치
        cv2.line(image, (LEFT_SHOULDER_X, LEFT_SHOULDER_Y),
                 (LEFT_ELBOW_X, LEFT_ELBOW_Y), COLOR, THCIKNESS, cv2.LINE_AA)

    if (LEFT_ELBOW_X != 0 and LEFT_WRIST_X != 0):
        # 왼쪽 팔꿈치 - 왼쪽 손목
        cv2.line(image, (LEFT_ELBOW_X, LEFT_ELBOW_Y),
                 (LEFT_WRIST_X, LEFT_WRIST_Y), COLOR, THCIKNESS, cv2.LINE_AA)
    if (CENTER_SHOULDER_X != 0 and CENTER_HIP_X != 0):
        # 중어꺠 - 중덩이
        cv2.line(image, (CENTER_SHOULDER_X, CENTER_SHOULDER_Y),
                 (CENTER_HIP_X, CENTER_HIP_Y), COLOR, THCIKNESS, cv2.LINE_AA)
    if (RIGHT_HIP_X != 0 and CENTER_HIP_X != 0):
        # 오른쪽 엉덩이 - 중덩이
        cv2.line(image, (RIGHT_HIP_X, RIGHT_HIP_Y),
                 (CENTER_HIP_X, CENTER_HIP_Y), COLOR, THCIKNESS, cv2.LINE_AA)
    if (LEFT_HIP_X != 0 and CENTER_HIP_X != 0):
        # 왼쪽 엉덩이 - 중덩이
        cv2.line(image, (LEFT_HIP_X, LEFT_HIP_Y),
                 (CENTER_HIP_X, CENTER_HIP_Y), COLOR, THCIKNESS, cv2.LINE_AA)
    if (RIGHT_HIP_X != 0 and RIGHT_KNEE_X != 0):
        # 오른쪽 엉덩이 - 오른쪽 무릎
        cv2.line(image, (RIGHT_HIP_X, RIGHT_HIP_Y),
                 (RIGHT_KNEE_X, RIGHT_KNEE_Y), COLOR, THCIKNESS, cv2.LINE_AA)
    if (RIGHT_KNEE_X != 0 and RIGHT_ANKLE_X != 0):
        # 오른쪽 무릎 - 오른쪽 발목
        cv2.line(image, (RIGHT_KNEE_X, RIGHT_KNEE_Y),
                 (RIGHT_ANKLE_X, RIGHT_ANKLE_Y), COLOR, THCIKNESS, cv2.LINE_AA)
    if (LEFT_HIP_X != 0 and LEFT_KNEE_X != 0):
        # 왼쪽 엉덩이 - 왼쪽 무릎
        cv2.line(image, (LEFT_HIP_X, LEFT_HIP_Y),
                 (LEFT_KNEE_X, LEFT_KNEE_Y), COLOR, THCIKNESS, cv2.LINE_AA)
    if (LEFT_KNEE_X != 0 and LEFT_ANKLE_X != 0):
        # 왼쪽 무릎 - 왼쪽 발목
        cv2.line(image, (LEFT_KNEE_X, LEFT_KNEE_Y),
                 (LEFT_ANKLE_X, LEFT_ANKLE_Y), COLOR, THCIKNESS, cv2.LINE_AA)

    return image
