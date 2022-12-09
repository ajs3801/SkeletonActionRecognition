# 인체 파트 별로 나눠서 처리하는 모델에 맞게 스켈레톤 추출

# 1. 왼쪽 상체 : 11, 12, 13, 15, 23 랜드마크 좌표 + 왼쪽 팔꿈치 관절 + 왼쪽 어깨 관절 => (22, )
# 2. 오른쪽 상체 : 11, 12, 14, 16, 24 랜드마크 좌표 + 오른쪽 팔꿈치 관절 + 오른쪽 어깨 관절 => (22, )
# 3. 왼쪽 하체 : 11, 23, 24, 25, 27 랜드마크 좌표 + 왼쪽 고관절 + 왼쪽 무릎 관절 => (22, )
# 4. 오른쪽 하체 : 12, 23, 24, 26, 28 랜드마크 좌표 + 오른쪽 고관절 + 오른쪽 무릎 관절 => (22, )

from numpy import angle
from configs import *
import extraFeatures

# video_path : 스켈레톤을 추출하려고 하는 비디오의 경로
# video_name : 비디오의 파일 이름, csv파일 이름을 비디오 이름과 통일해주기 위해서 필요함
# csv_path : 추출된 csv 파일이 저장될 경로
# degree : 동영상 rotation 각도


def getParts(temp, angles):
    # 왼쪽 상체, 오른쪽 상체, 왼쪽 하체, 오른쪽 하체 순서로 재구성
    left_upper = np.hstack(
        [temp[0:12], temp[16:20], temp[48:52], angles[0:1], angles[2:3]])
    right_upper = np.hstack(
        [temp[0:8], temp[12:16], temp[20:24], temp[52:56], angles[1:2], angles[3:4]])
    left_lower = np.hstack(
        [temp[0:4], temp[48:60], temp[64:68], angles[4:5], angles[6:7]])
    right_lower = np.hstack(
        [temp[4:8], temp[48:56], temp[60:64], temp[68:72], angles[5:6], angles[7:8]])
    temp2 = np.hstack(
        [left_upper, right_upper, left_lower, right_lower])
    return temp2


def extractPoseV3(action, video_path, video_name, csv_path, degree):
    # 프레임마다 뽑힌 스켈레톤 좌표를 하나로 모으기 위하여 비어있는 넘파이 배열 생성
    # 파트 하나당 feature 총 22개, 파트가 4개이므로 총 88개
    # 하나의 비디오 -> (30, 88)의 넘파이 배열로 추출
    extract = np.empty((1, 88))

    # 비디오 캡쳐 시작
    cap = cv2.VideoCapture(video_path)
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Action : {0} / filename : {1} / Skeleton Extraction Finished".format(
                    action, video_name))
                break

            if degree == 90:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif degree == 270:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # 미디어파이프를 이용하여 스켈레톤 추출
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # 모든 랜드마크의 피쳐를 1차원으로 펼쳐서 temp 변수에 저장
            temp = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_world_landmarks.landmark]).flatten(
            ) if results.pose_world_landmarks else np.zeros(132)

            # print(video_name)
            angles = extraFeatures.extractAngles(results)

            temp2 = getParts(temp, angles)

            extract = np.append(extract, [temp2], axis=0)

    # 첫번째 열은 아무 의미 없는 값이 들어가있기 때문에 지워줌
    extract = np.delete(extract, (0), axis=0)
    extract = extract.astype(np.float32)
    # print(extract.shape)

    # 30 프레임에서 추출한 관절 정보들을 하나의 csv 파일로 저장
    # 소수 다섯번째 자리까지만 저장
    out_path = os.path.join(csv_path, video_name) + '.csv'
    np.savetxt(out_path, extract, delimiter=",", fmt='%.5f')
    cap.release()


def main():
    # degrees = [0, 90, 270]
    degrees = [0]
    # 지정한 운동들을 스켈레톤 추출하여 저장
    '''for action in actions:
        for filename in os.listdir("./videos/model_test_videos/{0}".format(action)):
            if filename.endswith('.DS_Store'):
                continue
            video_path = os.path.join(
                "./videos/model_test_videos/{0}".format(action), filename)
            csv_path = os.path.join('./skeleton_csv/{0}_csv'.format(action))
            for degree in degrees:
                extractPoseV3(video_path, filename +
                              str(degree), csv_path, degree)

    for action in actions:
        for filename in os.listdir("./valid_videos/{0}".format(action)):
            if filename.endswith('.DS_Store'):
                continue
            video_path = os.path.join(
                "./valid_videos/{0}".format(action), filename)
            csv_path = os.path.join('./csv_part_valid/{0}_csv'.format(action))
            for degree in degrees:
                extractPoseV3(video_path, filename +
                              str(degree), csv_path, degree)'''

    for action in actions:
        for filename in os.listdir("./videos/model_test_videos/{0}".format(action)):
            if filename.endswith('.DS_Store'):
                continue
            video_path = os.path.join(
                "./videos/model_test_videos/{0}".format(action), filename)
            csv_path = os.path.join(
                './skeleton_csv/csv_modelv3_test/{0}_csv'.format(action))
            for degree in degrees:
                extractPoseV3(action, video_path, filename +
                              str(degree), csv_path, degree)


if __name__ == '__main__':
    main()
