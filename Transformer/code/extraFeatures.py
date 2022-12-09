import numpy as np

# ab, ac 벡터 사이 각도 구하기 - radian 값으로 리턴해줌


def getAngle(landmarks, a, b, c):
    A = landmarks[a]
    B = landmarks[b]
    C = landmarks[c]
    ab = B - A
    ac = C - A
    ab_len = np.linalg.norm(ab)
    ac_len = np.linalg.norm(ac)
    cos = np.dot(ab, ac) / (ab_len * ac_len)
    angle = np.array([np.arccos(cos)])
    return angle


def landmark2nparray(landmark):
    return np.array([landmark.x, landmark.y, landmark.z, landmark.visibility])


# input : mediapipe의 pose.process()의 results
# output : 8개의 관절 사이 각도로 이루어진 1차원 넘파이 배열 (8,)
def extractAngles(results):
    if not results.pose_world_landmarks:
        return np.zeros(8)
    temp = np.array(results.pose_world_landmarks.landmark)
    landmarks = np.array([landmark2nparray(x) for x in temp])
    out = getAngle(landmarks, 13, 11, 15)  # 왼쪽 팔꿈치 관절 0
    out = np.append(out, getAngle(landmarks, 14, 12, 16))  # 오른쪽 팔꿈치 관절 1
    out = np.append(out, getAngle(landmarks, 11, 13, 23))  # 왼쪽 어깨 관절 2
    out = np.append(out, getAngle(landmarks, 12, 14, 24))  # 오른쪽 어깨 관절 3
    out = np.append(out, getAngle(landmarks, 23, 11, 25))  # 왼쪽 고관절 4
    out = np.append(out, getAngle(landmarks, 24, 12, 26))  # 오른쪽 고관절 5
    out = np.append(out, getAngle(landmarks, 25, 23, 27))  # 왼쪽 무릎 관절 6
    out = np.append(out, getAngle(landmarks, 26, 24, 28))  # 오른쪽 무릎 관절 7
    # print(out.shape)
    return out
