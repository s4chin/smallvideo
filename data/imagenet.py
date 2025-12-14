# uint8 version of imagenet - https://github.com/cloneofsimo/imagenet.int8/pull/3
from torch.utils.data import Dataset
import numpy as np
import json


class ImageNet:
    def __init__(self, config):
        return

    def get_trainset(self, transform):
        return ImageNetDataset(
            "data_cache/inet.train.npy", "data_cache/inet.train.json", length=1276152, transform=transform
        )

    def get_valset(self, transform):
        return ImageNetDataset("data_cache/inet.val.npy", "data_cache/inet.val.json", length=5000, transform=transform)


class ImageNetDataset(Dataset):
    def __init__(self, data_path, labels_path=None, transform=None, length=None):
        assert length is not None, "length param cannot be None"
        self.data = np.memmap(data_path, dtype="uint8", mode="r", shape=(length, 4096))
        with open(labels_path, "r") as f:
            self.labels = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label, label_text = self.labels[idx]
        image = image.astype(np.float32).reshape(4, 32, 32)
        image = (image / 255.0 - 0.5) * 24.0 / 12.0
        # print(image.min(), image.max(), type(image))
        return image, label
