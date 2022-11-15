from configs import *
from model_v3 import Model

seq_length = config['seq_length']
data_dim = config['data_dim']

model_path = '/Users/jaejoon/LGuplus/main_project/lstm/model/model_mk10_20frame_8.5_100_0.0104_0.0005.pt'


def initModel():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(model_path, map_location=device)
    print(model)
    return model


def testModel(model, test_data):
    # start = time.time()
    model.eval()
    with torch.no_grad():
        out = model(test_data)
        out = np.squeeze(out)
        out = F.softmax(out, dim=0)
        # print(actions[out.numpy().argmax()])
    # m, s = divmod(time.time() - start, 60)
    # print(f'Inference time: {m:.0f}m {s:.5f}s')
    # print(out)
    # print(out.numpy().argmax())
    return out[out.numpy().argmax()], actions[out.numpy().argmax()]


def main():
    return 0


if __name__ == '__main__':
    main()
