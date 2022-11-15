import numpy as np
from configs import config


def random10():  # 0~9 중에 무작위로 5개 샘플링, 중복 없음
    a = np.random.choice(10, 5, replace=False)
    a.sort()
    return a


def random20():  # 0~29 중에 무작위로 20개 숫자를 샘플링, 중복 없음
    a = np.random.choice(30, 20, replace=False)
    a.sort()
    return a  # 20개


# def random20_padding():  # 0~29 중에 무작위로 20개 숫자를 샘플링, 중복 없음, 패딩 추가하여 30개 리턴
#     a = np.random.choice(30, 20, replace=False)
#     a.sort()
#     last = np.array([a[19]])
#     print(a)
#     print(last)
#     for i in range(10):  # 마지막 프레임을 10번 추가로 넣어줌
#         a = np.append(a, last)
#     print(a)
#     return a  # 30개


def random30():  # 0~29 중에 무작위로 30개 숫자를 샘플링, 중복 있음
    a = np.random.choice(30, 30, replace=True)
    a.sort()
    return a  # 30개


def seq20(start):  # start부터 start+19까지 숫자를 반환
    res = [x for x in range(30)]
    return res[start : start + 20]


def seq20_padding(start):  # start부터 start+19까지 숫자를 반환 + 마지막 숫자 10번 추가로 넣어줌 (패딩)
    res = [x for x in range(30)]
    res = res[start : start + 20]
    for i in range(10):
        res.append(start + 19)  # 마지막 프레임을 10번 추가로 넣어줌
    return res


# data : npy 파일(하나의 비디오)
# res : npy 파일에서 랜덤으로 샘플링한 20개의 프레임 데이터 (20,88)
def rand_sample20(data):
    frames = random20()
    res = np.empty((1, config["data_dim"]))
    for frame in frames:
        res = np.append(res, [data[frame]], axis=0)
    res = np.delete(res, (0), axis=0)
    res = res.astype(np.float16)
    return res


def seq_sample20(data, start):
    frames = seq20(start)
    res = np.empty((1, config["data_dim"]))
    for frame in frames:
        res = np.append(res, [data[frame]], axis=0)
    res = np.delete(res, (0), axis=0)
    res = res.astype(np.float16)
    return res


# 30프레임
def random_sample30(data):
    frames = random30()
    res = np.empty((1, config["data_dim"]))
    for frame in frames:
        res = np.append(res, [data[frame]], axis=0)
    res = np.delete(res, (0), axis=0)
    res = res.astype(np.float16)
    return res


# def randomSample20Padding(data):
#     frames = random20_padding()
#     res = np.empty((1, config["data_dim"]))
#     for frame in frames:
#         res = np.append(res, [data[frame]], axis=0)
#     res = np.delete(res, (0), axis=0)
#     res = res.astype(np.float32)
#     return res


def seq_sample20_padding(data, start):
    frames = seq20_padding(start)
    res = np.empty((1, config["data_dim"]))
    for frame in frames:
        res = np.append(res, [data[frame]], axis=0)
    res = np.delete(res, (0), axis=0)
    res = res.astype(np.float16)
    return res
