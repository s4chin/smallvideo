import numpy as np
import json
import sys
from pathlib import Path

data_path = Path(sys.argv[1])
labels_path = Path(sys.argv[2])

images = np.memmap(data_path, dtype="uint8", mode="r", shape=(1281152, 4096))
with open(labels_path, "r") as f:
    labels = json.load(f)

num_val_samples = 5000
num_samples = images.shape[0]
num_train_samples = num_samples - num_val_samples

train_images = images[:num_train_samples]
np.save(data_path.with_suffix(".train.npy"), train_images)
with open(labels_path.with_suffix(".train.json"), "w") as f:
    json.dump(labels[:num_train_samples], f)

val_images = images[num_train_samples:]
np.save(data_path.with_suffix(".val.npy"), val_images)
with open(labels_path.with_suffix(".val.json"), "w") as f:
    json.dump(labels[num_train_samples:], f)
