import torch
import torchinfo
import numpy as np
from torch.utils.data import DataLoader
from model_v3 import Model
from mydataset import MyDataset
from configs import config

# 모델 파라미터 설정
lstm_layers = config["lstm_layers"]
data_dim = config["data_dim"]
dropout = config["dropout"]
seq_length = config["seq_length"]

# 하이퍼 파라미터 설정
learning_rate = 0.0001
epochs = 200
batch_size = 1024
model_name = "Modelv3_{}".format(config["model_version"])

# 데이터 셋 설정
print("\n[Datasets]")
train_dataset_path = "./dataset/mydataset_v3_train_30frames_Nov1.npy"
validation_dataset_path = "./dataset/mydataset_v3_valid_30frames_Nov1.npy"
test_dataset_path = "./dataset/mydataset_v3_test_30frames_Nov1.npy"

train_dataset = MyDataset(train_dataset_path, "Training")
validation_dataset = MyDataset(validation_dataset_path, "Validation")
test_dataset = MyDataset(test_dataset_path, "Test")

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(
    dataset=validation_dataset, batch_size=batch_size, shuffle=True
)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# 모델 설정 / cpu, gpu 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model(device, learning_rate)
model.to(device)


def main():
    # 모델 구조 요약
    print("\n[Model Summary]")
    torchinfo.summary(model, (1, 1, seq_length, data_dim))

    # 모델 학습
    model.train_(epochs, train_loader, validation_loader, 25)

    # Best Valid Acc를 기록한 모델로 Test 진행
    model.restore()

    real_y, pred_y = model.predict(test_loader)
    correct = len(np.where(pred_y == real_y)[0])
    total = len(pred_y)
    test_acc = correct / total
    print("Test Accuracy (Top-1) at Best Epoch : %.4f" % (test_acc))

    # 학습 그래프
    model.plot()

    # 이미 학습된 모델을 불러와서 테스트 진행할 경우
    # model_path = (
    #     "/Users/jaejoon/SkeletonActionRecognition/LSTM/code/best_model/Modelv3_mk11.pt"
    # )
    # test_model = Model(device)
    # test_model.load_state_dict(torch.load(model_path, device))
    # real_y, pred_y = test_model.predict(test_loader)
    # correct = len(np.where(pred_y == real_y)[0])
    # total = len(pred_y)
    # test_acc = correct / total
    # print("Test Accuracy of Model : %.4f" % (test_acc))


if __name__ == "__main__":
    main()
