import numpy as np
from configs import config

seq_length = config["seq_length"]
data_dim = config["data_dim"]

# 동영상 하나에 해당하는 파일을 입력으로 받음 (30, 88)
# x y z 좌표, visibility, 관절 각도에 따라 각각 알맞은 노이즈를 추가한 파일을 반환
# 노이즈는 정규 분포에서 샘플링한 값을 더해줌
def add_noise(file):
    upper_left = file[:, 0:22]
    upper_right = file[:, 22:44]
    lower_left = file[:, 44:66]
    lower_right = file[:, 66:88]
    parts = [upper_left, upper_right, lower_left, lower_right]
    new_parts = []

    for part in parts:
        xyz1 = part[:, 0:3] + np.random.normal(0, 0.05, (seq_length, 3))
        xyz2 = part[:, 4:7] + np.random.normal(0, 0.05, (seq_length, 3))
        xyz3 = part[:, 8:11] + np.random.normal(0, 0.05, (seq_length, 3))
        xyz4 = part[:, 12:15] + np.random.normal(0, 0.05, (seq_length, 3))
        xyz5 = part[:, 16:19] + np.random.normal(0, 0.05, (seq_length, 3))

        v1 = part[:, 3:4] + np.random.normal(0, 0.02, (seq_length, 1))
        v2 = part[:, 7:8] + np.random.normal(0, 0.02, (seq_length, 1))
        v3 = part[:, 11:12] + np.random.normal(0, 0.02, (seq_length, 1))
        v4 = part[:, 15:16] + np.random.normal(0, 0.02, (seq_length, 1))
        v5 = part[:, 19:20] + np.random.normal(0, 0.02, (seq_length, 1))

        v1 = np.clip(v1, 0, 1)
        v2 = np.clip(v2, 0, 1)
        v3 = np.clip(v3, 0, 1)
        v4 = np.clip(v4, 0, 1)
        v5 = np.clip(v5, 0, 1)

        angles = part[:, -2:] + np.random.normal(0, 0.02, (seq_length, 2))
        angles = np.clip(angles, 0, 3.1415)

        new_part = np.hstack([xyz1, v1, xyz2, v2, xyz3, v3, xyz4, v4, xyz5, v5, angles])
        new_parts.append(new_part)

    noise_file = np.hstack(new_parts)
    return noise_file
