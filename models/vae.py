import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else self.in_channels

        self.blocks = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=self.in_channels),
            nn.SiLU(),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=self.out_channels),
            nn.SiLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        )

        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
    
    def forward(self, x):
        h = self.blocks(x)
        if self.in_channels != self.out_channels:
            x = self.shortcut(x)
        return x + h

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(Downsample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else self.in_channels
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(Upsample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class VAEEncoder(nn.Module):
    def __init__(self, in_channels, base_channels, num_res_blocks, channel_mults, z_channels):
        super(VAEEncoder, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.z_channels = z_channels

        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.blocks = nn.ModuleList()
        in_channel_mults = [1] + list(channel_mults)
        for i, channel_mult in enumerate(channel_mults):
            ch_in = base_channels * in_channel_mults[i]
            ch_out = base_channels * channel_mults[i]

            block_in = ch_in
            block_out = ch_out
            for block in range(num_res_blocks):
                self.blocks.append(ResBlock(block_in, block_out))
                block_in = block_out
            
            if i < len(channel_mults) - 1:
                self.blocks.append(Downsample(block_in))
        
        self.blocks.append(ResBlock(block_in, block_out))
        self.blocks.append(nn.GroupNorm(num_groups=32, num_channels=block_out))
        self.blocks.append(nn.SiLU())
        self.blocks.append(nn.Conv2d(block_out, 2 * z_channels, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.blocks:
            x = block(x)
        return x

class VAEDecoder(nn.Module):
    def __init__(self, out_channels, base_channels, num_res_blocks, channel_mults, z_channels):
        super(VAEDecoder, self).__init__()
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_res_blocks = num_res_blocks
        self.channel_mults = channel_mults
        self.z_channels = z_channels

        self.conv_in = nn.Conv2d(z_channels, self.base_channels * channel_mults[-1], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList()
        channel_mults_rev = list(channel_mults[::-1])
        in_channel_mults_rev = [channel_mults[-1]] + list(channel_mults_rev)
        for i, channel_mult in enumerate(channel_mults_rev):
            ch_in = self.base_channels * in_channel_mults_rev[i]
            ch_out = self.base_channels * channel_mults_rev[i]

            block_in = ch_in
            block_out = ch_out
            for block in range(num_res_blocks + 1):
                self.blocks.append(ResBlock(block_in, block_out))
                block_in = block_out
            
            if i < len(channel_mults_rev) - 1:
                self.blocks.append(Upsample(block_in))
        
        self.blocks.append(ResBlock(block_in, block_out))
        self.blocks.append(nn.GroupNorm(num_groups=32, num_channels=block_out))
        self.blocks.append(nn.SiLU())
        self.blocks.append(nn.Conv2d(block_out, self.out_channels, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.conv_in(x)
        for block in self.blocks:
            x = block(x)
        return x

class DiagonalGaussian(nn.Module):
    def __init__(self, sample_z = False):
        super(DiagonalGaussian, self).__init__()
        self.sample_z = sample_z
    
    def forward(self, z):
        mean, logvar = z.chunk(2, dim=1)
        if self.sample_z:
            std = torch.exp(logvar * 0.5)
            return mean + torch.randn_like(mean) * std
        else:
            return mean

class VAE(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels, num_res_blocks, channel_mults, z_channels, sample_z = False, scale_factor = 1.0, shift_factor = 0.0):
        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_res_blocks = num_res_blocks
        self.channel_mults = channel_mults
        self.z_channels = z_channels
        self.sample_z = sample_z

        self.encoder = VAEEncoder(self.in_channels, self.base_channels, self.num_res_blocks, self.channel_mults, self.z_channels)
        self.decoder = VAEDecoder(self.out_channels, self.base_channels, self.num_res_blocks, self.channel_mults, self.z_channels)
        self.sampler = DiagonalGaussian(sample_z=self.sample_z)

        self.scale_factor = scale_factor
        self.shift_factor = shift_factor
    
    def encode(self, x):
        return (self.sampler(self.encoder(x)) - self.shift_factor) / self.scale_factor
    
    def decode(self, z):
        return self.decoder((z * self.scale_factor) + self.shift_factor)
    
    def forward(self, x):
        return self.decode(self.encode(x))

if __name__ == "__main__":
    vae_encoder = VAEEncoder(in_channels=3, base_channels=32, num_res_blocks=2, channel_mults=[1, 2, 4, 8], z_channels=16)
    img = torch.randn(1, 3, 256, 256)
    z = vae_encoder(img)
    print(f"{z.shape=}")

    sampler = DiagonalGaussian(sample_z = True)
    z = sampler(z)
    print(f"{z.shape=}")

    vae_decoder = VAEDecoder(out_channels=3, base_channels=32, num_res_blocks=2, channel_mults=[1, 2, 4, 8], z_channels=16)
    x_reconstructed = vae_decoder(z)
    print(f"{x_reconstructed.shape=}")

    for res in [256, 512, 1024]:
        img = torch.randn(1, 3, res, res)
        vae = VAE(in_channels=3, out_channels=3, base_channels=32, num_res_blocks=2, channel_mults=[1, 2, 4, 8], z_channels=16, scale_factor=1.0, shift_factor=0.0)
        x_reconstructed = vae(img)
        print(f"{x_reconstructed.shape=}")