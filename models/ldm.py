import torch
import torch.nn as nn


class TimestepScheduler:
    def add_noise(self, x, t, noise):
        assert t.shape[0] == x.shape[0]
        if t.ndim == 1:
            t = t.reshape(t.shape[0], 1, 1, 1, 1)
        x_t = (1 - t) * noise + t * x
        return x_t

    def get_timestep(self, b):
        return torch.rand(b)


class LatentDiffusionModel(nn.Module):
    """
    The latent part will be implemented later
    """
    def __init__(self, dit, vae): # some config
        super().__init__()
        self.scheduler = TimestepScheduler()
        self.dit = dit
        self.vae = vae


    def prep_data(self, batch):
        # timestep sample, vae encode/decode, etc.
        batch["t"] = self.scheduler.get_timestep(batch["x"].shape[0])
        return batch


    def compute_loss(self, batch, noise, prediction):
        x = batch["x"]
        target = x - noise

        return nn.functional.mse_loss(prediction, target)


    def forward(self, batch, is_train=False):
        t = batch["t"]
        x = batch["x"]
        conds = {}

        noise = torch.randn_like(x)
        x_t = self.scheduler.add_noise(x, t, noise)

        prediction = self.dit(x_t, t, conds)
        return prediction, noise
    
    def fwd_bwd_one_step(self, batch):
        batch = self.prep_data(batch)
        prediction, noise = self.forward(batch, is_train=True)
        loss = self.compute_loss(batch=batch, noise=noise, prediction=prediction)
        loss.backward()
        return loss.item()



if __name__ == "__main__":
    scheduler = TimestepScheduler()
    print(scheduler)
