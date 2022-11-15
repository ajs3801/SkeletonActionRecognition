import numpy as np

# 나중에 좌표말고 다른 피쳐 넣을때 쓸 함수들


def landmark2xyz(landmark):
    return np.array([landmark.x, landmark.y, landmark.z])


def t(p, q, r):
    x = p-q
    return np.dot(r-q, x)/np.dot(x, x)

# p-q line with joint r


def jldistance(p, q, r):
    p = landmark2xyz(p)
    q = landmark2xyz(q)
    r = landmark2xyz(r)
    return np.linalg.norm(t(p, q, r)*(p-q)+q-r)


def jjdistance(l1, l2):
    p1 = landmark2xyz(l1)
    p2 = landmark2xyz(l2)
    squared_dist = np.sum((p1-p2)**2, axis=0)
    dist = np.sqrt(squared_dist)
    return dist


# print(temp[44:])
    # print(temp[60:63] - temp[63:66])
    # print(temp[60:63])
    # print(landmark2xyz(results.pose_world_landmarks.landmark[15]))
    # print(jjdistance(temp[60:63], temp[63:66]))
    # print(jldistance(
    #    results.pose_world_landmarks.landmark[12], results.pose_world_landmarks.landmark[14], results.pose_world_landmarks.landmark[16]))
