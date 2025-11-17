from omegaconf import OmegaConf
import argparse
import importlib


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    trainer_config = config.trainer
    module_path, class_name = trainer_config.target.rsplit(".", 1)
    trainer_class = getattr(importlib.import_module(module_path), class_name)
    trainer = trainer_class(config)
    trainer.train() if hasattr(trainer, "train") else trainer.train_cifar10()
