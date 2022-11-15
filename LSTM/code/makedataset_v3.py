import numpy as np
import os
import sampling20 as smp
import noise
from configs import config, actions


# 라벨 인덱스
def make_label(action):
    idx = actions.index(action)
    return np.array([idx], dtype=np.float16)


# training, validation, test 데이터 셋 별로 각각의 클래스에 대한 npy 파일들이 존재
# 동일한 데이터 셋의 동일한 클래스를 가진 데이터들을 하나로 합침
# 시퀀스 샘플링과 노이즈 추가
def make_action_dataset(action, dataset_type):
    print("\n\nDataset Type :", dataset_type)
    print("action :", action)
    dataset = np.empty(
        (1, config["seq_length"] * config["data_dim"] + 1), dtype=np.float16
    )

    cnt = 0
    dirpath = "./skeleton_npy/npy_modelv3_{}/{}_npy".format(dataset_type, action)
    filenames = os.listdir(dirpath)
    num_files = len(filenames)

    for filename in filenames:
        print(action, cnt, "/", num_files, "|", filename)

        cnt += 1

        if not filename.endswith(".npy"):
            continue

        file = np.load(os.path.join(dirpath, filename))

        # 연속된 20 프레임 + 마지막 프레임 10번 패딩
        # 5번 샘플링
        # 시작 인덱스는 랜덤
        starts = smp.random10()
        for start in starts:
            sample_file = smp.seq_sample20_padding(file, start)
            noise_sample_file = noise.add_noise(sample_file)  # 노이즈 추가

            sample_file = sample_file.flatten()
            noise_sample_file = noise_sample_file.flatten()

            sample_file = np.concatenate([sample_file, make_label(action)])
            noise_sample_file = np.concatenate([noise_sample_file, make_label(action)])

            dataset = np.append(dataset, [sample_file], axis=0)
            dataset = np.append(dataset, [noise_sample_file], axis=0)

        # 20 프레임 랜덤 샘플링 + 랜덤 패딩 10 프레임
        # 5번 샘플링
        for i in range(5):
            sample_file = smp.random_sample30(file)
            noise_sample_file = noise.add_noise(sample_file)  # 노이즈 추가

            sample_file = sample_file.flatten()
            noise_sample_file = noise_sample_file.flatten()

            sample_file = np.concatenate([sample_file, make_label(action)])
            noise_sample_file = np.concatenate([noise_sample_file, make_label(action)])

            dataset = np.append(dataset, [sample_file], axis=0)
            dataset = np.append(dataset, [noise_sample_file], axis=0)

    dataset = np.delete(dataset, (0), axis=0)

    print("Dataset Type :", dataset_type)
    print("Actions :", actions)
    print("Sequence Length :", config["seq_length"])
    print("# of Data :", dataset.shape[0])

    dataset_path = "./dataset/mydataset_v3_{}_{}_{}frames_Nov1.npy".format(
        dataset_type, action, config["seq_length"]
    )

    dataset = dataset.astype(np.float16)
    np.save(dataset_path, dataset)


# 각각의 데이터셋에 대하여 클래스 단위로 나눠져있는 npy 파일들을 하나로 합침
def make_complete_dataset(dataset_type):
    dataset = np.empty((1, config["seq_length"] * config["data_dim"] + 1))

    for action in actions:
        file = np.load(
            "./dataset/mydataset_v3_{}_{}_{}frames_Oct31.npy".format(
                dataset_type, action, config["seq_length"]
            )
        )

        dataset = np.append(dataset, file, axis=0)

    dataset = np.delete(dataset, (0), axis=0)

    dataset_path = "./dataset/mydataset_v3_{}_{}frames_Oct31.npy".format(
        dataset_type, config["seq_length"]
    )
    dataset = dataset.astype(np.float16)
    np.save(dataset_path, dataset)

    print("Dataset Type :", dataset_type)
    print("Actions :", actions)
    print("Sequence Length :", config["seq_length"])
    print("# of Data :", dataset.shape[0])


def main():
    dataset_types = ["train", "valid", "test"]

    for dataset_type in dataset_types:
        for action in actions:
            make_action_dataset(action, dataset_type)

    for dataset_type in dataset_types:
        make_complete_dataset(dataset_type)


if __name__ == "__main__":
    main()
