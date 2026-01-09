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

# Shuffle indices with fixed seed for reproducibility
rng = np.random.default_rng(seed=42)
indices = rng.permutation(num_samples)
train_indices = indices[:num_train_samples]
val_indices = indices[num_train_samples:]

train_out = np.memmap(
    data_path.with_suffix(".train.npy"),
    dtype="uint8",
    mode="w+",
    shape=(num_train_samples, 4096)
)
train_out[:] = images[train_indices]
train_out.flush()
del train_out
with open(labels_path.with_suffix(".train.json"), "w") as f:
    json.dump([labels[i] for i in train_indices], f)

val_out = np.memmap(
    data_path.with_suffix(".val.npy"),
    dtype="uint8",
    mode="w+",
    shape=(num_val_samples, 4096)
)
val_out[:] = images[val_indices]
val_out.flush()
del val_out
with open(labels_path.with_suffix(".val.json"), "w") as f:
    json.dump([labels[i] for i in val_indices], f)
