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


    def prep_data(self, batch, p_drop_cond=0.):
        # timestep sample, vae encode/decode, etc.
        batch["t"] = self.scheduler.get_timestep(batch["x"].shape[0])
        conds = batch.get("label", None)
        if conds is not None:
            if conds.ndim == 1:
                conds = conds.reshape(-1, 1)
            conds = 1. + conds
            cond_drop_mask = torch.rand_like(conds) < p_drop_cond
            conds = torch.where(cond_drop_mask, torch.zeros_like(conds), conds)
            batch["label"] = conds.to(dtype=torch.long)
        return batch


    def compute_loss(self, batch, is_train=False):
        p_drop_cond = 0.1 if is_train else 0.0
        batch = self.prep_data(batch, p_drop_cond=p_drop_cond)
        prediction, noise = self.forward(batch, is_train=is_train)
        x = batch["x"]
        target = x - noise

        return nn.functional.mse_loss(prediction, target)


    def forward(self, batch, is_train=False):
        t = batch["t"]
        x = batch["x"]
        conds = {
            "label": batch["label"]
        }

        noise = torch.randn_like(x)
        x_t = self.scheduler.add_noise(x, t, noise)

        prediction = self.dit(x_t, t, conds)
        return prediction, noise
    
    def sample(self, batch):
        batch = self.prep_data(batch)
        x = batch["x"]
        conds = {
            "label": batch["label"]
        }
        x_t = torch.randn_like(x)

        num_steps = 10
        timesteps = torch.linspace(1., 0., num_steps + 1)
        for i, t in enumerate(timesteps[:-1]):
            t = torch.ones_like(batch["t"]) * t
            velocity = self.dit(x_t, t, conds)
            x_t = x_t + velocity * (timesteps[i+1] - timesteps[i])
        return x_t



if __name__ == "__main__":
    scheduler = TimestepScheduler()
    print(scheduler)
