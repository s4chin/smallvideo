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
    def __init__(self):
        # dataloader, scheduler, dit setup
        trainset = torchvision.datasets.CIFAR10(root="./data_cache", train=True, transform=transform, download=True)
        valset = torchvision.datasets.CIFAR10(root="./data_cache", train=False, transform=transform, download=True)
        self.train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
        self.val_loader = torch.utils.data.DataLoader(valset, batch_size=4, shuffle=False, num_workers=2)

        dit = DiT(3, 32, 8, 4, (1, 1, 1), 4, 256, 0)

        self.ldm = LatentDiffusionModel(dit, vae=None)
        self.optimizer = optim.AdamW(dit.parameters(), lr=1e-4)

    def train_cifar10(self, num_iters=5):
        for step, data in enumerate(self.train_loader):
            if step >= num_iters:
                print("num_iters reached")
                exit(0)
            images, labels = data

            batch = {}
            batch["x"] = images.unsqueeze(2)
            batch["conds"] = labels

            loss = self.ldm.fwd_bwd_one_step(batch)
            print(loss)
            self.optimizer.step()




if __name__ == "__main__":
    trainer = LDMTrainer()
    trainer.train_cifar10(num_iters=50)
