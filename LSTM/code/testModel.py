from configs import *
from model_v3 import Model
import Draw

data_dim = config['data_dim']
output_dim = config['output_dim']
seq_length = config['seq_length']

font = cv2.FONT_HERSHEY_SIMPLEX


def initModel(model_name):
    model_path = os.path.join(
        '/Users/jaejoon/LGuplus/main_project/lstm/model/modelv3', model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(model_path, map_location=device)
    # print(model)
    print('\nModel Initialization Succeeded!\n')
    return model


def testModel(model, test_data):
    model.eval()
    with torch.no_grad():
        out = model(test_data)
        out = np.squeeze(out)
        out = F.softmax(out, dim=0)

    # 가장 높은 확률과 가장 높은 확률인 클래스를 리턴
    return out[out.numpy().argmax()], actions[out.numpy().argmax()]


def test(model_name):
    # 테스트 결과 confusion matrix
    confusion_matrix = np.zeros((output_dim, output_dim), dtype=int)

    # 클래스 별 accuracy
    class_accuracy = np.zeros(output_dim)

    # 전체 accuracy
    test_accuracy = None

    # 모델 초기화
    model = initModel(model_name)

    # 여러가지 모델 비교하기 위한 데이터용 파일
    test_output_file = open(
        './test_log/model_test_output_log.csv', 'a', newline='')

    # 임의의 모델이 어떤 비디오에 대해 어떻게 예측했는지 확인하기 위한 파일
    model_log_file = open('./test_log/model_log/{}_{}frame.csv'.format(
        model_name, seq_length), 'w', newline='')
    test_wr = csv.writer(test_output_file)
    model_wr = csv.writer(model_log_file)

    # 테스트 데이터셋 불러옴
    testset_file_20frame = '/Users/jaejoon/LGuplus/main_project/lstm/dataset/mydataset_v3_test_20frames.csv'
    testset_20frame = np.loadtxt(
        testset_file_20frame, delimiter=',', dtype=np.float32)
    testset_file_30frame = '/Users/jaejoon/LGuplus/main_project/lstm/dataset/mydataset_v3_test_30frames.csv'
    testset_30frame = np.loadtxt(
        testset_file_30frame, delimiter=',', dtype=np.float32)

    if seq_length == 20:
        for i in range(len(testset_20frame)):
            model_input = testset_20frame[i, :-config['output_dim']]
            model_input = torch.from_numpy(model_input)
            label = testset_20frame[i, -config['output_dim']:]

            prob, action = testModel(model, model_input)

            confusion_matrix[np.argmax(label), actions.index(action)] += 1
            model_wr.writerow([actions[np.argmax(label)], action])

    elif seq_length == 30:
        for i in range(len(testset_30frame)):
            model_input = testset_30frame[i, :-config['output_dim']]
            model_input = torch.from_numpy(model_input)
            label = testset_30frame[i, -config['output_dim']:]

            prob, action = testModel(model, model_input)

            confusion_matrix[np.argmax(label), actions.index(action)] += 1
            model_wr.writerow([actions[np.argmax(label)], action])

    np.savetxt('./test_log/confusion_matrix/{}_{}frame_confusion_matrix.csv'.format(model_name, seq_length),
               confusion_matrix, delimiter=",", fmt='%.5f')

    right_inference_cnt = 0  # 클래스 상관없이 전체에서 몇개 맞췄는지 카운트
    testset_size = 0  # 전체 테스트 데이터셋 사이즈
    for i in range(output_dim):
        class_accuracy[i] = confusion_matrix[i, i] / \
            np.sum(confusion_matrix[i])  # 클래스 별 accuracy
        testset_size += np.sum(confusion_matrix[i])
        right_inference_cnt += confusion_matrix[i, i]
    test_accuracy = right_inference_cnt / testset_size  # 전체 테스트 데이터셋 accuracy

    test_wr.writerow([model_name, '{}frames'.format(seq_length), test_accuracy,
                     class_accuracy[0], class_accuracy[1], class_accuracy[2], class_accuracy[3], class_accuracy[4], class_accuracy[5], class_accuracy[6], class_accuracy[7], class_accuracy[8]])

    model_log_file.close()
    test_output_file.close()
    print('Model Test Complete!')


def main():
    # 테스트할 모델 파일들이 있는 디렉토리
    models_path = '/Users/jaejoon/LGuplus/main_project/lstm/model/modelv3'
    # 여러 모델에 대해서 테스트 수행
    for model_name in os.listdir(models_path):
        if not model_name.endswith('.pt'):
            continue
        print('\nModel Name : ' + model_name)
        test(model_name)


if __name__ == '__main__':
    main()
