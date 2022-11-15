import torch
import numpy as np
from torch.utils.data import Dataset
from configs import config

output_dim = config["output_dim"]


class MyDataset(Dataset):
    def __init__(self, dataset_file, dataset_name):
        data = np.load(dataset_file)
        self.len = data.shape[0]
        print("# of {} Data : {}".format(dataset_name, self.len))
        self.x_data = torch.from_numpy(data[:, 0:-output_dim])
        self.y_data = torch.from_numpy(data[:, -output_dim:])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
