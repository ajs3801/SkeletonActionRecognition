from configs import *
import tensorflow as tf
import extraFeatures
import count
from count import action_count_length
from extract_v3 import getParts
import Draw


data_dim = config['data_dim']
seq_length = config['seq_length']

font = cv2.FONT_HERSHEY_SIMPLEX


def initModel(model_name):
    model =tf.keras.models.load_model('Transformer/code/model_v3.0.0')
    print(model)
    print('\nModel Initialization Succeeded!\n')
    return model


def testModel(model, test_data):
    # start = time.time()
    # model.eval()
    #test_data = np.reshape(test_data, (-1, 1760,1))
    out = model.predict(np.reshape(test_data, (-1, 2640,1)))

    # print(actions[out.numpy().argmax()])
    # m, s = divmod(time.time() - start, 60)
    # print(f'Inference time: {m:.0f}m {s:.5f}s')
    # print(out)
    # print(out.numpy().argmax())
    res = np.argmax(out)
    return out[0][res], actions[res]


# 비디오를 인풋으로 받아서 카운트한 액션을 순서대로 리스트에 넣어서 반환해줌
def getActionSequence(video, model):
    extract = np.empty((1, data_dim))
    action_count = []
    action_sequence = []

    _fpss = np.array([])

    cap = cv2.VideoCapture(video)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        i = 0
        sum = 0
        SFPS = ""
        while cap.isOpened():
            start_t = timeit.default_timer()
            success, image = cap.read()
            if not success:
                print(video)
                if video == 0:
                    continue
                else:
                    break

            # 미디어파이프를 이용하여 스켈레톤 추출
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            try:
                Draw.DrawSkeleton(
                    image, results.pose_landmarks.landmark, (203, 192, 255))
            # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            except:
                pass

            temp = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_world_landmarks.landmark]).flatten(
            ) if results.pose_world_landmarks else np.zeros(132)

            angles = extraFeatures.extractAngles(results)

            temp2 = getParts(temp, angles)
            extract = np.append(extract, [temp2], axis=0)
            extract = extract.astype(np.float32)

            image = cv2.flip(image, 1)

            if(extract.shape[0] > seq_length):
                extract = np.delete(extract, (0), axis=0)

                prob, action = testModel(model, tf.convert_to_tensor(extract))
                prob = prob.item()

                if prob > config['threshold']:
                    action_count.append(action)
                    if len(action_count) > action_count_length:
                        action_count.pop(0)

                    res, detected_action = count.countAction(action_count)
                    if res:
                        action_sequence.append(detected_action)

            #         cv2.putText(image, action, (50, 100),
            #                     font, 2, (0, 0, 255), 3)
            #         cv2.putText(image, str(prob), (50, 300),
            #                     font, 2, (0, 0, 255), 3)
            # cv2.putText(image, 'squat : {}, lunge : {}, pushup : {}'.format(count.cnt['squat'], count.cnt['lunge'], count.cnt['pushup']),
            #             (50, 400), font, 2, (0, 255, 0), 2)

            # fps 계산
            terminate_t = timeit.default_timer()
            FPS = 1./(terminate_t - start_t)
            sum = FPS+sum
            if SFPS == "":
                SFPS = str(int(FPS))
            if i == 10:
                sum = sum/10
                sum = round(sum, 4)
                SFPS = str(int(sum))
                i = 0
                sum = 0
            i += 1
            # cv2.putText(image, "FPS : "+SFPS, (800, 100),
            #             font, 2, (255, 0, 0), 3)
            _fpss = np.append(_fpss, [float(SFPS)])
            try:
                Draw.DrawText(image, str(
                    action), count.cnt['squat'], count.cnt['lunge'], count.cnt['pushup'], SFPS)
            except:
                pass
            cv2.imshow('Testing Engine', image)
            cv2.waitKey(1)

    cap.release()
    _avg_fps = np.average(_fpss)
    return action_sequence, _avg_fps


def getTestLabel(filename):
    label = filename[-7:-4]
    label = list(label)
    label.insert(0, label[0])
    label.insert(2, label[2])
    label.insert(4, label[4])
    return label


def test(model_name):
    # 총 테스트 비디오 개수
    test_videos_num = 0
    # 테스트 비디오 중 올바르게 예측한 비디오 개수
    positive = 0

    # 모델 초기화
    model = initModel(model_name)

    # 여러가지 모델 비교하기 위한 데이터용 파일
    test_file = open('test2.csv', 'a', newline='')
    # 임의의 모델이 어떤 비디오에 대해 어떻게 예측했는지 확인하기 위한 파일
    model_file = open('{}_{}frame_test2.csv'.format(
        model_name, seq_length), 'a', newline='')
    test_wr = csv.writer(test_file)
    model_wr = csv.writer(model_file)

    # 동영상 별 평균 fps 저장하기위한 리스트
    fpss = np.array([])

    # 테스트 비디오 위치
    test_videos_path = 'main_project/lstm/Mix'
    for filename in os.listdir(test_videos_path):
        # 운동 카운트 리셋
        count.resetCnt()

        result = None
        test_videos_num += 1

        video_path = os.path.join(test_videos_path, filename)
        # 비디오 파일명에서 라벨 추출
        label = getTestLabel(filename)

        # 비디오에서 액션 시퀀스, fps 추출
        predicted, fps = getActionSequence(video_path, model)
        fpss = np.append(fpss, [fps])

        print('label :', label)
        print('model :', predicted)

        # 추출한 액션 시퀀스가 라벨과 동일한지 비교
        if predicted == label:
            positive += 1
            result = 'True'
            print('=> True\n')
        else:
            result = 'False'
            print('=> False\n')

        # 비디오 파일명, fps, 모델이 예측한 액션 시퀀스, 맞았는지 틀렸는지 여부
        model_wr.writerow([filename, int(fps), predicted, result])

    # 모델의 평균 fps
    avg_fps = int(np.average(fpss))
    # 전체 동영상 중 맞춘 동영상 비율
    accuracy = int((positive/test_videos_num) * 100)

    print('test accuracy : {}%'.format(accuracy))
    test_wr.writerow([model_name, str(accuracy)+'%',
                     str(avg_fps)+'fps', seq_length])

    model_file.close()
    test_file.close()


def main():

    # 웹캠으로 테스트하고 싶을 때
    model_name = 'model_mk10_20frame_8.8_300_0.0005_0.0001.pt'
    model = initModel(model_name)
    #cap = cv2.VideoCapture('sub_project/Transformer/202208021434_original_SLP.avi')
    #s.StartEngine(cap)
    #cap.release(
    predicted, fps = getActionSequence('Transformer/code/data/202208021434_original_SLP.avi', model)
    label = getTestLabel('Transformer/code/data/202208021434_original_SLP.avi')
    print('label :', label)
    print('model :', predicted)

    # 테스트할 모델 파일들이 있는 디렉토리
    # models_path = '/Users/jaejoon/LGuplus/main_project/lstm/model/for_test'
    # # 여러 모델에 대해서 테스트 수행
    # for model_name in os.listdir(models_path):
    #     print('\n'+model_name)
    #     # 현재 테스트 중인 모델에 몇 프레임이 들어가고 있는지 표시
    #     print('sequence length :', seq_length, '\n')
    # test(model_name)


if __name__ == '__main__':
    main()
