import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import time
import os
from configs import config, actions

# 왼쪽 팔, 오른쪽 팔, 왼쪽 다리, 오른쪽 다리 각각 나눠서 처리하는 모델

layers = config["lstm_layers"]
data_dim = config["data_dim"]
seq_length = config["seq_length"]
dropout = config["dropout"]
model_name = "Modelv3_{}".format(config["model_version"])


class Model(nn.Module):
    def __init__(self, device="cpu", learning_rate=0.0001):
        super(Model, self).__init__()

        self.lstm1 = nn.LSTM(
            22, 10, num_layers=layers, batch_first=True, bias=True, dropout=dropout
        )
        self.lstm2 = nn.LSTM(
            22, 10, num_layers=layers, batch_first=True, bias=True, dropout=dropout
        )
        self.lstm3 = nn.LSTM(
            22, 10, num_layers=layers, batch_first=True, bias=True, dropout=dropout
        )
        self.lstm4 = nn.LSTM(
            22, 10, num_layers=layers, batch_first=True, bias=True, dropout=dropout
        )
        self.lstm5 = nn.LSTM(
            20, 10, num_layers=layers, batch_first=True, bias=True, dropout=dropout
        )
        self.lstm6 = nn.LSTM(
            20, 10, num_layers=layers, batch_first=True, bias=True, dropout=dropout
        )
        self.lstm7 = nn.LSTM(
            20, 10, num_layers=layers, batch_first=True, bias=True, dropout=dropout
        )
        self.fc1 = nn.Linear(10, 10, bias=True)
        self.fc2 = nn.Linear(10, len(actions), bias=True)
        self.silu = nn.SiLU()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = device

    def forward(self, x):
        x = x.view([-1, seq_length, data_dim])
        x = x.float()

        left_upper_data = x[:, :, 0:22]
        right_upper_data = x[:, :, 22:44]
        left_lower_data = x[:, :, 44:66]
        right_lower_data = x[:, :, 66:88]

        left_upper, _ = self.lstm1(left_upper_data)
        right_upper, _ = self.lstm2(right_upper_data)
        left_lower, _ = self.lstm3(left_lower_data)
        right_lower, _ = self.lstm4(right_lower_data)

        upper = torch.cat([left_upper, right_upper], dim=2)
        lower = torch.cat([left_lower, right_lower], dim=2)

        upper, _ = self.lstm5(upper)
        lower, _ = self.lstm6(lower)

        body = torch.cat([upper, lower], dim=2)
        body, _ = self.lstm7(body)
        x = self.silu(self.fc1(body[:, -1, :]))
        x = self.fc2(x)
        return x

    def train_(self, epochs, train_loader, validation_loader, save):
        self.train_loss_val = []
        # self.valid_loss_val = []
        self.train_acc_val = []
        self.valid_acc_val = []

        best_acc = -1
        best_epoch = -1

        since = time.time()

        print("\nModel Name :", model_name)
        print("Model will be trained on {}\n".format(self.device))

        for epoch in range(1, epochs + 1):
            print("[Epoch {:3d} / {}]".format(epoch, epochs))
            epoch_start = time.time()
            self.train()
            epoch_loss = 0.0
            total = 0
            correct = 0
            for batch_idx, (data, target) in enumerate(
                tqdm.tqdm(train_loader, desc="Training")
            ):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.forward(data)
                target = target.long()  # cross entropy loss 입력은 long으로 넣어줘야함
                target = target.squeeze()  # [batch size, 1] 크기가 되어서 발생하는 오류 해결

                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

            epoch_end = time.time()
            m, s = divmod(epoch_end - epoch_start, 60)
            epoch_loss /= len(train_loader)
            train_acc = correct / total
            self.train_acc_val.append(train_acc)

            with torch.no_grad():
                self.eval()
                real_y, pred_y = self.predict(validation_loader)
                correct = (pred_y == real_y).sum().item()
                total = len(pred_y)
                valid_acc = correct / total
                self.valid_acc_val.append(valid_acc)

                # for batch_idx, (data, target) in enumerate(validation_loader):
                #     data = data.to(self.device)
                #     target = target.to(self.device)
                #     target = target.long()
                #     target = target.squeeze()
                #     valid_output = self.forward(data)
                #     valid_loss = self.criterion(valid_output, target)

            print(
                "Training Loss = {:.4f} | Training Accuracy = {:.4f} | Validation Accuracy = {:.4f}".format(
                    epoch_loss, train_acc, valid_acc
                )
            )
            print(f"Training Time: {m:.0f}m {s:.0f}s\n")

            if best_acc < valid_acc:
                print(
                    "=> Best Model Updated : Epoch = {}, Validation Accuracy = {:.4f}\n".format(
                        epoch, valid_acc
                    )
                )
                best_acc = valid_acc
                best_epoch = epoch
                torch.save(self.state_dict(), "./best_model/{}.pt".format(model_name))
            else:
                print()

            self.train_loss_val.append(loss.item())
            # self.valid_loss_val.append(valid_loss.item())

            if (epoch % save) == 0:
                torch.save(
                    self.state_dict(),
                    "./model/modelv3/{}_{}_{:.4f}.pt".format(
                        model_name, epoch, loss.item()
                    ),
                )

        torch.save(self.state_dict(), "./model/{}_final.pt".format(model_name))

        m, s = divmod(time.time() - since, 60)
        print("\nTraining Finished...!!")
        print("\nBest Valid acc : %.2f at epoch %d" % (best_acc, best_epoch))
        print(f"Total Time: {m:.0f}m {s:.0f}s\nModel was trained on {self.device}!")

    def restore(self):
        with open(os.path.join("./best_model/Modelv3_mk11.pt"), "rb") as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)

    def predict(self, data_loader):
        self.eval()
        correct_y = []
        pred_y = []
        with torch.no_grad():
            for batch_data in data_loader:
                batch_x, batch_y = batch_data
                pred = self.forward(batch_x.to(self.device))
                _, predicted = torch.max(pred.data, 1)
                correct_y.append(batch_y.numpy())
                pred_y.append(predicted.cpu().numpy())
        correct_y = np.concatenate(correct_y, axis=0).squeeze()
        pred_y = np.concatenate(pred_y, axis=0)
        return correct_y, pred_y

    def plot(self):
        plt.plot(np.array(self.train_loss_val), "b")
        plt.plot(np.array(self.train_acc_val), "r")
        plt.plot(np.array(self.valid_acc_val), "g")
        plt.savefig("graph.png")
        plt.show()
