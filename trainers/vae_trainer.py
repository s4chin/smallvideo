import torch
import torch.nn as nn
import os
from models.vae import VAE


class VAETrainer:
    def __init__(self, config):
        self.config = config

        self.vae = VAE(self.config.vae)

        self.optimizer = torch.optim.Adam(
            self.vae.parameters(), lr=self.config.train.learning_rate
        )
        self.num_training_steps = self.config.train.num_training_steps
        self.log_train_loss_steps = self.config.train.log_train_loss_steps
        self.eval_every_n_steps = self.config.train.eval_every_n_steps
        self.eval_steps = self.config.train.eval_steps
        self.save_every_n_steps = self.config.train.save_every_n_steps
        self.beta = self.config.train.beta

        self.train_dataloader = None
        self.val_dataloader = None

    def mse_loss(self, x, x_recon):
        return nn.functional.mse_loss(x, x_recon, reduction="mean")

    def kl_loss(self, mean, logvar):
        return -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

    def save(self, step):
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(self.vae.state_dict(), f"checkpoints/vae_{step}.pth")

    def single_fwd_bwd_pass(self, batch):
        x = batch["img"]
        assert x.ndim == 4, f"images should be of shape (b, c, h, w), but got {x.shape}"

        x_recon, mean, logvar = self.vae(x, return_mean_logvar=True)

        mse_loss = self.mse_loss(x, x_recon)
        kl_loss = self.kl_loss(mean, logvar)
        loss = mse_loss + self.beta * kl_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return mse_loss.item(), kl_loss.item()

    def validation(self):
        val_mse_losses = []
        val_kl_losses = []

        self.vae.eval()
        for step in range(self.eval_steps):
            batch = next(iter(self.val_dataloader))

            with torch.no_grad():
                x_recon, mean, logvar = self.vae(batch["img"], return_mean_logvar=True)

            mse_loss = self.mse_loss(batch["img"], x_recon)
            kl_loss = self.kl_loss(mean, logvar)

            val_mse_losses.append(mse_loss.item())
            val_kl_losses.append(kl_loss.item())

        avg_val_mse_loss = sum(val_mse_losses) / len(val_mse_losses)
        avg_val_kl_loss = sum(val_kl_losses) / len(val_kl_losses)
        self.vae.train()
        return avg_val_mse_loss, avg_val_kl_loss

    def train(self):
        self.vae.train()
        for step in range(self.num_training_steps):
            batch = next(iter(self.train_dataloader))
            mse_loss, kl_loss = self.single_fwd_bwd_pass(batch)
            if step % self.log_train_loss_steps == 0:
                print(
                    f"Step {step+1}/{self.num_training_steps}, Training MSE Loss: {mse_loss}, Training KL Loss: {kl_loss}"
                )
            if step % self.eval_every_n_steps == 0:
                val_mse_loss, val_kl_loss = self.validation()
                print(
                    f"Step {step+1}/{self.num_training_steps}, Validation MSE Loss: {val_mse_loss}, Validation KL Loss: {val_kl_loss}"
                )
            if step % self.save_every_n_steps == 0:
                self.save(step)
        self.vae.eval()
