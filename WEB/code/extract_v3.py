# 인체 파트 별로 나눠서 처리하는 모델에 맞게 스켈레톤 추출

# 1. 왼쪽 상체 : 11, 12, 13, 15, 23 랜드마크 좌표 + 왼쪽 팔꿈치 관절 + 왼쪽 어깨 관절 => (22, )
# 2. 오른쪽 상체 : 11, 12, 14, 16, 24 랜드마크 좌표 + 오른쪽 팔꿈치 관절 + 오른쪽 어깨 관절 => (22, )
# 3. 왼쪽 하체 : 11, 23, 24, 25, 27 랜드마크 좌표 + 왼쪽 고관절 + 왼쪽 무릎 관절 => (22, )
# 4. 오른쪽 하체 : 12, 23, 24, 26, 28 랜드마크 좌표 + 오른쪽 고관절 + 오른쪽 무릎 관절 => (22, )

import numpy as np
import cv2
import os
import random
from configs import mp_pose, actions
import extrafeatures

# video_path : 스켈레톤을 추출하려고 하는 비디오의 경로
# video_name : 비디오의 파일 이름, npy파일 이름을 비디오 이름과 통일해주기 위해서 필요함
# npy_path : 추출된 npy 파일이 저장될 경로
# degree : 동영상 rotation 각도


def get_parts(skeleton, angles):
    # 왼쪽 상체, 오른쪽 상체, 왼쪽 하체, 오른쪽 하체 순서로 재구성
    left_upper = np.hstack(
        [skeleton[0:12], skeleton[16:20], skeleton[48:52], angles[0:1], angles[2:3]]
    )
    right_upper = np.hstack(
        [
            skeleton[0:8],
            skeleton[12:16],
            skeleton[20:24],
            skeleton[52:56],
            angles[1:2],
            angles[3:4],
        ]
    )
    left_lower = np.hstack(
        [skeleton[0:4], skeleton[48:60], skeleton[64:68], angles[4:5], angles[6:7]]
    )
    right_lower = np.hstack(
        [
            skeleton[4:8],
            skeleton[48:56],
            skeleton[60:64],
            skeleton[68:72],
            angles[5:6],
            angles[7:8],
        ]
    )
    parts = np.hstack([left_upper, right_upper, left_lower, right_lower])
    return parts  # (88, )


# frame을 degree만큼 회전시킨 후 반환
def rotate_frame(frame, degree):
    height, width, _ = frame.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    dst = cv2.warpAffine(frame, matrix, (width, height))
    return dst


# 왼쪽으로 shear한 프레임 반환
def shear_left_frame(frame):
    h, w, c = frame.shape

    pts1 = np.float32([[0, 0], [0, h], [w, h], [w, 0]])
    pts2 = np.float32([[0, 100], [0, h - 100], [w, h], [w, 0]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (w, h))

    return result


# 오른쪽으로 shear한 프레임 반환
def shear_right_frame(frame):
    h, w, c = frame.shape

    pts1 = np.float32([[0, 0], [0, h], [w, h], [w, 0]])
    pts2 = np.float32([[0, 0], [0, h], [w, h - 100], [w, 100]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (w, h))

    return result


def extract_pose_v3(action, video_path, video_name, npy_path, augmentation_option):
    # 프레임마다 뽑힌 스켈레톤 좌표를 하나로 모으기 위하여 비어있는 넘파이 배열 생성
    # 파트 하나당 feature 총 22개, 파트가 4개이므로 총 88개
    # 하나의 비디오 -> (30, 88)의 넘파이 배열로 추출
    extract = np.empty((1, 88))

    # 비디오 캡쳐 시작
    cap = cv2.VideoCapture(video_path)
    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print(
                    "Action : {0} | Filename : {1} | Augmentation : {2} | Skeleton Extraction Finished".format(
                        action, video_name, augmentation_option
                    )
                )
                break

            # augmentation_option에 따라서 이미지 변환한 후 미디어 파이프로 스켈레톤 추출
            if augmentation_option == "flip":  # 좌우반전
                frame = cv2.flip(frame, 1)
            elif augmentation_option == "rotate_5":
                frame = rotate_frame(frame, 5)
            elif augmentation_option == "rotate_-5":
                frame = rotate_frame(frame, -5)
            elif augmentation_option == "rotate_10":
                frame = rotate_frame(frame, 10)
            elif augmentation_option == "rotate_-10":
                frame = rotate_frame(frame, -10)
            elif augmentation_option == "shear_left":
                frame = shear_left_frame(frame)
            elif augmentation_option == "shear_right":
                frame = shear_right_frame(frame)
            else:
                pass  # 원본 비디오

            # 미디어파이프를 이용하여 스켈레톤 추출
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_skeleton = pose.process(frame)

            # 모든 랜드마크의 피쳐를 1차원으로 펼쳐서 temp 변수에 저장
            np_skeleton = (
                np.array(
                    [
                        [res.x, res.y, res.z, res.visibility]
                        for res in mp_skeleton.pose_world_landmarks.landmark
                    ]
                ).flatten()
                if mp_skeleton.pose_world_landmarks
                else np.zeros(132, dtype=np.float32)
            )

            # 관절 각도 계산
            angles = extrafeatures.extractAngles(mp_skeleton)

            # 모델 입력으로 사용하기 위한 포맷에 맞게 변환
            parts = get_parts(np_skeleton, angles)

            # extract 변수에 붙임
            extract = np.append(extract, [parts], axis=0)

    # 첫번째 열은 아무 의미 없는 값이 들어가있기 때문에 지워줌
    extract = np.delete(extract, (0), axis=0)
    extract = extract.astype(np.float16)

    # 30 프레임에서 추출한 관절 정보들을 하나의 npy 파일로 저장
    # 원래 파일 이름뒤에 augmentation option 붙여서 저장
    out_name = video_name + "_" + augmentation_option + ".npy"
    out_path = os.path.join(npy_path, out_name)
    np.savetxt(out_path, extract)
    cap.release()


# videos : 비디오 파일명들의 리스트
# 비디오들을 train : valid : test = 8 : 1 : 1로 분할하여 반환
# 반환되는 리스트에는 해당하는 비디오들의 파일명들이 들어있음
def split_video(videos):
    dataset_size = len(videos)
    train_size = int(dataset_size * 0.8)
    validation_size = int(dataset_size * 0.1)

    random.shuffle(videos)

    train_videos = videos[0:train_size]
    valid_videos = videos[train_size : train_size + validation_size]
    test_videos = videos[train_size + validation_size : dataset_size]

    return train_videos, valid_videos, test_videos


def main():
    # 원본을 포함하여 8가지 augmentation
    augmentation_options = [
        "original",
        "flip",
        "rotate_5",
        "rotate_-5",
        "rotate_10",
        "rotate_-10",
        "shear_left",
        "shear_right",
    ]

    # 지정한 운동들을 스켈레톤 추출하여 저장
    for action in actions:
        # train, valid, test로 분할
        videos = os.listdir("./videos/original_videos/{0}".format(action))
        train_videos, valid_videos, test_videos = split_video(videos)

        # training video 추출
        print("\nTraing Videos\n")
        for video in train_videos:
            if not video.endswith(".avi"):
                continue
            video_path = os.path.join(
                "./videos/original_videos/{0}".format(action), video
            )
            npy_path = os.path.join(
                "./skeleton_npy/npy_modelv3_train/{0}_npy".format(action)
            )
            for aug_opt in augmentation_options:
                extract_pose_v3(action, video_path, video, npy_path, aug_opt)

        # validation video 추출
        print("\nValidation Videos\n")
        for video in valid_videos:
            if not video.endswith(".avi"):
                continue
            video_path = os.path.join(
                "./videos/original_videos/{0}".format(action), video
            )
            npy_path = os.path.join(
                "./skeleton_npy/npy_modelv3_valid/{0}_npy".format(action)
            )
            for aug_opt in augmentation_options:
                extract_pose_v3(action, video_path, video, npy_path, aug_opt)

        # test video 추출
        print("\nTest Videos\n")
        for video in test_videos:
            if not video.endswith(".avi"):
                continue
            video_path = os.path.join(
                "./videos/original_videos/{0}".format(action), video
            )
            npy_path = os.path.join(
                "./skeleton_npy/npy_modelv3_test/{0}_npy".format(action)
            )
            for aug_opt in augmentation_options:
                extract_pose_v3(action, video_path, video, npy_path, aug_opt)


if __name__ == "__main__":
    main()
