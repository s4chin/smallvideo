import os

import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from models import DiT, LatentDiffusionModel

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.RandomHorizontalFlip(0.5),
])


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if not dist.is_initialized() and world_size > 1:
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    return local_rank, rank, world_size


def setup_fsdp(model, local_rank, world_size):
    if world_size == 1 or not torch.cuda.is_available():
        return model

    return FSDP(model)


class LDMTrainer:
    def __init__(self, config):
        # dataloader, scheduler, dit setup
        self.config = config
        self.trainer_config = self.config.trainer

        self.local_rank, self.rank, self.world_size = setup_distributed()

        trainset = torchvision.datasets.CIFAR10(
            root="./data_cache",
            train=True,
            transform=transform,
            download=True,
        )
        valset = torchvision.datasets.CIFAR10(
            root="./data_cache",
            train=False,
            transform=transform,
            download=True,
        )

        self.train_sampler = None
        self.val_sampler = None
        if self.world_size > 1:
            self.train_sampler = torch.utils.data.DistributedSampler(
                trainset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=False,
            )
            self.val_sampler = torch.utils.data.DistributedSampler(
                valset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=True,
            )

        train_loader_kwargs = {
            "batch_size": 16,
            "num_workers": 2,
            "pin_memory": True,
        }
        if self.train_sampler is not None:
            train_loader_kwargs["sampler"] = self.train_sampler
        else:
            train_loader_kwargs["shuffle"] = True

        val_loader_kwargs = {
            "batch_size": 16,
            "num_workers": 2,
            "pin_memory": True,
        }
        if self.val_sampler is not None:
            val_loader_kwargs["sampler"] = self.val_sampler
        else:
            val_loader_kwargs["shuffle"] = False

        self.train_loader = torch.utils.data.DataLoader(trainset, **train_loader_kwargs)
        self.val_loader = torch.utils.data.DataLoader(valset, **val_loader_kwargs)

        dit = DiT(**config.dit)

        self.ldm = LatentDiffusionModel(dit, vae=None)

        self.device = f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu"

        self.ldm.to(self.device)
        self.ldm = setup_fsdp(self.ldm, self.local_rank, self.world_size)

        self.optimizer = optim.AdamW(
            self.ldm.parameters(),
            lr=self.trainer_config.learning_rate
        )

    def train_cifar10(self, num_iters=None, val_every=None):
        num_iters = num_iters or self.trainer_config.num_iters
        val_every = val_every or self.trainer_config.val_every

        def prep_batch(data):
            images, labels = data

            batch = {}
            batch["x"] = images.unsqueeze(2).to(self.device)
            batch["label"] = labels.to(self.device)
            return batch

        epoch = 0
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        for step, data in enumerate(self.train_loader):
            if step > 0 and (step + 1) % val_every == 0:
                self.ldm.eval()
                val_losses = []
                for val_step, val_data in enumerate(self.val_loader):
                    num_val_iters = self.trainer_config.num_val_iters
                    if val_step >= num_val_iters:
                        break
                    val_batch = prep_batch(val_data)
                    with torch.no_grad():
                        val_loss = self.ldm.compute_loss(val_batch)
                    val_losses.append(val_loss.item())
                print(f"Val loss: {sum(val_losses)/len(val_losses)}")

                # Sampling during validation
                for val_data in self.val_loader:
                    val_batch = prep_batch(val_data)
                    with torch.no_grad():
                        val_sample = self.ldm.sample(val_batch)
                    print(f"Val sample: {val_sample.shape}")
                    b, c, f, h, w = val_sample.shape
                    for i in range(min(2, b)):
                        sample = val_sample[i, :, 0, :, :]
                        sample = (sample * 0.5 + 0.5) * 255
                        sample = sample.permute(1, 2, 0)
                        sample = sample.detach().cpu().numpy().astype(np.uint8)
                        Image.fromarray(sample).save(f"step_{step}_sample_{i}.png")
                    break
                self.ldm.train()

            if step >= num_iters:
                print("num_iters reached")
                break

            batch = prep_batch(data)

            self.optimizer.zero_grad()
            loss = self.ldm.compute_loss(batch)
            loss.backward()
            self.optimizer.step()
            print(f"Train loss: {loss.item()}")

        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("configs/cifar10.yaml")
    trainer = LDMTrainer(config)
    trainer.train_cifar10(num_iters=100)
