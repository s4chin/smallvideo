import torch.nn as nn
import torch

from einops import rearrange

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False



def scale_shift(x, scale=None, shift=None):
    bs, seq_len, num_channels = x.shape
    device = x.device

    if scale is None:
        scale = torch.zeros(bs, num_channels, device=device)
    if shift is None:
        shift = torch.zeros(bs, num_channels, device=device)
    
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Patchify(nn.Module):
    def __init__(self, in_channels, num_channels, patch_size=(1, 1, 1)):
        super().__init__()

        self.patch_size = patch_size
        self.p_f, self.p_h, self.p_w = patch_size
        self.linear = nn.Linear(in_channels * self.p_f * self.p_h * self.p_w, num_channels)

    def forward(self, x):
        # shape: x [b, c, f, h, w]

        bs, c, f, h, w = x.shape

        n_f, n_h, n_w = f // self.p_f, h // self.p_h, w // self.p_w

        x_patched = rearrange(x, "b c (n_f p_f) (n_h p_h) (n_w p_w) -> b (n_f n_h n_w) (c p_f p_h p_w)", n_f=n_f, n_h=n_h, n_w=n_w)

        return self.linear(x_patched)



class MHSA(nn.Module):
    def __init__(self, num_channels, num_heads):
        super().__init__()
        assert num_channels % num_heads == 0, "num_channels must be divisible by num_heads"
        self.qkv = nn.Linear(num_channels, num_channels * 3)
        self.proj = nn.Linear(num_channels, num_channels)

        self.num_heads = num_heads

    def forward(self, x):
        bs, seq_len, num_channels = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        if FLASH_ATTN_AVAILABLE:
            # Flash Attention expects (batch, seq_len, num_heads, head_dim)
            q = rearrange(q, "bs seq_len (n_head n_channels) -> bs seq_len n_head n_channels", n_head=self.num_heads)
            k = rearrange(k, "bs seq_len (n_head n_channels) -> bs seq_len n_head n_channels", n_head=self.num_heads)
            v = rearrange(v, "bs seq_len (n_head n_channels) -> bs seq_len n_head n_channels", n_head=self.num_heads)
            
            head_dim = num_channels // self.num_heads
            softmax_scale = 1.0 / (head_dim ** 0.5)
            attn_out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=softmax_scale, causal=False)
            
            attn_out = rearrange(attn_out, "bs seq_len n_head n_channels -> bs seq_len (n_head n_channels)")
        else:
            # Fallback to regular attention
            q = rearrange(q, "bs seq_len (n_head n_channels) -> bs n_head seq_len n_channels", n_head=self.num_heads)
            k = rearrange(k, "bs seq_len (n_head n_channels) -> bs n_head seq_len n_channels", n_head=self.num_heads)
            v = rearrange(v, "bs seq_len (n_head n_channels) -> bs n_head seq_len n_channels", n_head=self.num_heads)

            attn_logits = q @ k.transpose(-1, -2) / ((num_channels / self.num_heads) ** 0.5)
            attn_weights = attn_logits.softmax(dim=-1)

            attn_out = attn_weights @ v

            attn_out = rearrange(attn_out, "bs n_head seq_len n_channels -> bs seq_len (n_head n_channels)")

        return self.proj(attn_out)


class Mlp(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.linear = nn.Linear(num_channels, num_channels)
    
    def forward(self, x):
        return self.linear(x)


class DiTBlock(nn.Module):
    def __init__(self, num_channels, num_heads):
        super().__init__()

        self.mhsa_norm = nn.LayerNorm(num_channels, elementwise_affine=False)
        self.mhsa = MHSA(num_channels, num_heads)
        self.mlp_norm = nn.LayerNorm(num_channels, elementwise_affine=False)
        self.mlp = Mlp(num_channels)

        self.adaLN_mlp = nn.Linear(num_channels, 6 * num_channels) # can make it more complex


    def forward(self, x, cond):
        # shapes
        # x [bs, seq_len, num_channels], cond [bs, num_channels]

        gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2 = self.adaLN_mlp(cond).chunk(6, dim=-1)

        x_mhsa = scale_shift(self.mhsa_norm(x), scale=gamma_1, shift=beta_1)
        x_mhsa = self.mhsa(x_mhsa)
        x_mhsa = scale_shift(x_mhsa, scale=alpha_1)
        x = x + x_mhsa

        x_mlp = scale_shift(self.mlp_norm(x), scale=gamma_2, shift=beta_2)
        x_mlp = self.mlp(x_mlp)
        x_mlp = scale_shift(x_mlp, scale=alpha_2)
        x = x + x_mlp

        return x



class DiT(nn.Module):
    def __init__(self, num_channels, depth, num_heads):
        super().__init__()
        

        self.dit_blocks = nn.ModuleList([DiTBlock(num_channels, num_heads) for _ in range(depth)])
        self.final_layer = nn.Linear(num_channels, num_channels) # TBA

        return
    


    def forward(self, x, conds):
        # shapes
        # x [bs, f, c, h, w]

        x = self.patchify(x) # shape [bs, seq_len, num_channels]

        for block in self.dit_blocks:
            x = block(x)
        
        x = self.final_layer(x)

        x = self.unpatchify(x) # TBA

        return x


def test_patchify(bs, c, f, h, w, num_channels):
    patchify = Patchify(c, num_channels, patch_size=(1, 2, 2))
    x = torch.randn(bs, c, f, h, w)
    x_patched = patchify(x)
    print(f"{x.shape=}, {x_patched.shape=}")


def test_mhsa(bs, seq_len, num_channels, num_heads):
    mhsa = MHSA(num_channels, num_heads)
    x = torch.randn(bs, seq_len, num_channels)
    output_x = mhsa(x)
    assert output_x.shape == x.shape, f"Output shape {output_x.shape} does not match input shape {x.shape}"


def test_dit_block(bs, seq_len, num_channels, num_heads):
    block = DiTBlock(num_channels, num_heads)
    x = torch.randn(bs, seq_len, num_channels)
    cond = torch.randn(bs, num_channels)

    out = block(x, cond)
    assert out.shape == x.shape, f"Output shape {out.shape} does not match input shape {x.shape}"



if __name__ == '__main__':
    bs, c, f, h, w = 3, 16, 8, 32, 32
    num_channels = 32
    test_patchify(bs, c, f, h, w, num_channels)

    bs, seq_len, num_channels, num_heads = 3, 64, 32, 4

    test_mhsa(bs, seq_len, num_channels, num_heads)
    test_dit_block(bs, seq_len, num_channels, num_heads)
    pass