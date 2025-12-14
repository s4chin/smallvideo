import torchvision


class CIFAR10:
    def __init__(self, config):
        return

    def get_trainset(self, transform):
        return torchvision.datasets.CIFAR10(
            root="./data_cache",
            train=True,
            transform=transform,
            download=True,
        )

    def get_valset(self, transform):
        return torchvision.datasets.CIFAR10(
            root="./data_cache",
            train=False,
            transform=transform,
            download=True,
        )
