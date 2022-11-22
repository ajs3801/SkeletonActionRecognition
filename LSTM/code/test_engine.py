import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import extrafeatures
import draw
import StartEngine as s
import timeit
from configs import config, actions, mp_pose
from extract_v3 import get_parts
from model_v3 import Model
from count import Counter

data_dim = config["data_dim"]
seq_length = config["seq_length"]
action_count_threshold = config["action_count_threshold"]

font = cv2.FONT_HERSHEY_SIMPLEX


class Engine:
    def __init__(self, model_path, window_size, prob_threshold, count_threshold):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Model(self.device)
        self.model.load_state_dict(torch.load(model_path, self.device))
        self.window_size = window_size
        self.prob_threshold = prob_threshold
        self.count_threshold = count_threshold
        self.counter = Counter(count_threshold)

        print("\nEngine Init Completed\n")

    def print_engine_config(self):
        print("Device :", self.device)
        print("Model :")
        print(self.model)
        print("Window Size :", self.window_size)
        print("Prob Threshold :", self.prob_threshold)
        print("Count Threshold :", self.count_threshold)

    # 입력 : 모델, 테스트 데이터
    # 출력 : 확률, 클래스 이름
    # 추론 시간 계산 코드는 주석 처리되었음
    def test_model(self, test_data):
        # start = time.time()
        self.model.eval()
        with torch.no_grad():
            out = self.model(test_data)
            out = np.squeeze(out)
            out = F.softmax(out, dim=0)
            # print(actions[out.numpy().argmax()])
        # m, s = divmod(time.time() - start, 60)
        # print(f'Inference time: {m:.0f}m {s:.5f}s')
        predicted_label = out.numpy().argmax()
        return out[predicted_label], actions[predicted_label]

    # 하나의 프레임에서 추출한 스켈레톤을 모델 입력에 맞는 형태로 변환
    def to_model_input(self, mp_results):
        results = None  # 미디어파이프 추출 결과를 넘파이 배열로 정리
        if mp_results.pose_world_landmarks:
            results = np.array(
                [
                    [res.x, res.y, res.z, res.visibility]
                    for res in mp_results.pose_world_landmarks.landmark
                ]
            ).flatten()
        else:
            results = np.zeros(132)  # 추출 결과가 없을 경우 전부 0으로 처리

        # 관절 사이 각도 계산
        angles = extrafeatures.extractAngles(mp_results)

        # 관절 좌표와 관절 각도를 모델 입력에 맞는 형식으로 변환
        model_input = get_parts(results, angles)

        return model_input

    # 입력 : 비디오, 모델
    # 출력 : 카운트된 운동의 시퀀스, 운동 별 카운트 개수 ,평균 fps 반환
    def test_engine(self, video):
        model_inputs = np.empty((1, data_dim))
        action_window = []  # 최근 self.window_size개 만큼의 인식된 운동 클래스를 보는 윈도우
        counted_action_sequence = []  # 비디오에서 카운트된 운동의 시퀀스

        fpss = np.array([])

        cap = cv2.VideoCapture(video)

        with mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as pose:
            i = 0
            sum = 0
            SFPS = ""
            while cap.isOpened():
                start_t = timeit.default_timer()
                success, image = cap.read()
                if not success:
                    # print(video)
                    if video == 0:  # 웹캠 입력일 경우
                        continue
                    else:  # 비디오 입력일 경우
                        break

                # 미디어파이프를 이용하여 스켈레톤 추출
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                try:  # 화면에 스켈레톤 표시
                    draw.draw_skeleton(
                        image, results.pose_landmarks.landmark, (203, 192, 255)
                    )
                except:
                    pass

                image = cv2.flip(image, 1)

                # 미디어파이프 추출 결과를 모델 입력에 맞게 변환 후 윈도우에 넣어줌
                model_input = self.to_model_input(results)
                model_inputs = np.append(model_inputs, [model_input], axis=0)
                model_inputs = model_inputs.astype(np.float32)

                # 모델이 처리할 수 있는 시퀀스 길이만큼 시퀀스가 윈도우에 쌓였는지 체크
                if len(model_inputs) > seq_length:
                    model_inputs = np.delete(
                        model_inputs, (0), axis=0
                    )  # 새로운 시퀀스 들어오면 맨 처음 시퀀스 버림

                    # 모델 추론
                    prob, action = self.test_model(torch.Tensor(model_inputs))
                    prob = prob.item()

                    # prob_threshold를 넘었을 때만 액션(운동 클래스)이 인식된 것으로 처리
                    if prob > self.prob_threshold:
                        action_window.append(action)
                    else:
                        action_window.append(None)

                    # action window가 설정한 window 사이즈에 도달했는지 체크
                    if len(action_window) > self.window_size:
                        action_window.pop(0)  # 새로운 액션 들어오면 맨 처음 액션 버림
                        counted_action = self.counter.count(action_window)
                        if counted_action != None:
                            counted_action_sequence.append(counted_action)

                # fps 계산
                terminate_t = timeit.default_timer()
                FPS = 1.0 / (terminate_t - start_t)
                sum = FPS + sum
                if SFPS == "":
                    SFPS = str(int(FPS))
                if i == 10:
                    sum = sum / 10
                    sum = round(sum, 4)
                    SFPS = str(int(sum))
                    i = 0
                    sum = 0
                i += 1
                fpss = np.append(fpss, [float(SFPS)])

                # 현재 운동 카운트 화면에 표시
                try:
                    draw.draw_text(
                        image,
                        str(action),
                        self.counter.cnt["squat"],
                        self.counter.cnt["lunge"],
                        self.counter.cnt["pushup"],
                        SFPS,
                    )
                except:
                    pass

                cv2.imshow("Engine Test", image)
                cv2.waitKey(1)

        cap.release()
        avg_fps = np.average(fpss)

        return counted_action_sequence, self.counter.cnt, int(avg_fps)


def get_label(video):
    label = video[:-7]
    return label


def main():
    # 엔진에 탑재될 모델
    model_path = (
        "/Users/jaejoon/SkeletonActionRecognition/LSTM/code/best_model/Modelv3_mk11.pt"
    )
    # 엔진 생성, 파라미터 설정
    my_engine = Engine(
        model_path, window_size=10, prob_threshold=0.7, count_threshold=8
    )
    # 엔진 설정값 보여줌
    my_engine.print_engine_config()

    # 엔진 테스트
    test_videos = os.listdir("./videos/new_engine_test_videos")
    num_videos = len(test_videos)
    fpss = np.array([])
    correct = 0
    miss_videos = []
    for test_video in test_videos:
        if not test_video.endswith(".mp4"):
            continue
        # print(test_video)
        label = test_video[:-7]
        test_video_path = os.path.join("./videos/new_engine_test_videos", test_video)
        action_sequence, counts, fps = my_engine.test_engine(test_video_path)
        fpss = np.append(fpss, [fps])

        if len(action_sequence) == 1:
            if label == action_sequence[0]:
                correct += 1
        elif label == "stand" and len(action_sequence) == 0:
            correct += 1
        else:
            miss_videos.append(test_video)

        my_engine.counter.reset_cnt()

    avg_fps = np.average(fpss)
    acc = correct / num_videos

    print("total videos :", num_videos)
    print("acc : %.4f" % (acc))
    print("average fps :", int(avg_fps))
    print("miss counted videos :", miss_videos)


if __name__ == "__main__":
    main()
