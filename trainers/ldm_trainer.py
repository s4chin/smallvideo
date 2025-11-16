import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models import DiT, LatentDiffusionModel

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.RandomHorizontalFlip(0.5),
])


class LDMTrainer:
    def __init__(self, config):
        # dataloader, scheduler, dit setup
        self.config = config
        self.trainer_config = self.config.trainer

        trainset = torchvision.datasets.CIFAR10(root="./data_cache", train=True, transform=transform, download=True)
        valset = torchvision.datasets.CIFAR10(root="./data_cache", train=False, transform=transform, download=True)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)
        self.val_loader = torch.utils.data.DataLoader(valset, batch_size=16, shuffle=False, num_workers=2)

        dit = DiT(**config.dit)

        self.ldm = LatentDiffusionModel(dit, vae=None)
        self.optimizer = optim.AdamW(
            dit.parameters(),
            lr=self.trainer_config.learning_rate
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def train_cifar10(self, num_iters=None, val_every=None):
        num_iters = num_iters or self.trainer_config.num_iters
        val_every = val_every or self.trainer_config.val_every

        def prep_batch(data):
            images, labels = data

            batch = {}
            batch["x"] = images.unsqueeze(2)
            batch["label"] = labels
            return batch

        for step, data in enumerate(self.train_loader):
            if step > 0 and (step + 1) % val_every == 0:
                self.ldm.eval()
                for val_step, val_data in enumerate(self.val_loader):
                    num_val_iters = self.trainer_config.num_val_iters
                    if val_step >= num_val_iters:
                        break
                    val_batch = prep_batch(val_data)
                    with torch.no_grad():
                        val_loss = self.ldm.compute_loss(val_batch)
                    print(f"Val loss: {val_loss.item()}")
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


if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("configs/cifar10.yaml")
    trainer = LDMTrainer(config)
    trainer.train_cifar10(num_iters=100)
