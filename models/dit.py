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


def sincos_embedding_1d(t, half_dim):
    # Formula: PE_(pos,2i) = sin(pos/10000^(2i/d_model))
    #          PE_(pos,2i+1) = cos(pos/10000^(2i/d_model))
    # where i goes from 0 to half_dim-1
    if t.ndim == 1:
        t = t.unsqueeze(1) # [b, 1]
    b = t.shape[0]
    freqs = (10000 ** ((2. * torch.arange(0, half_dim)) / (2 * half_dim)))
    pe_sin = torch.sin(t / freqs)
    pe_cos = torch.cos(t / freqs)

    pe = torch.stack([pe_sin, pe_cos], dim=-1).reshape(b, 2 * half_dim)

    return pe


class TimeEmbedder(nn.Module):
    def __init__(self, t_dim, num_channels):
        super().__init__()
        self.t_dim = t_dim
        self.proj = nn.Sequential(
            nn.Linear(t_dim, num_channels),
            nn.SiLU(),
            nn.Linear(num_channels, num_channels),
        )
    
    def forward(self, t):
        t_emb = sincos_embedding_1d(t, self.t_dim // 2)
        return self.proj(t_emb)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, num_channels):
        super().__init__()
        self.label_embedding = nn.Embedding(1 + num_classes, num_channels)
    
    def forward(self, labels):
        # Assumption: 0 is for unconditional, other labels start at 1
        return self.label_embedding(labels)


class Patchify(nn.Module):
    def __init__(self, in_channels, num_channels, patch_size=(1, 1, 1)):
        super().__init__()

        self.patch_size = patch_size
        self.pf, self.ph, self.pw = patch_size
        self.linear = nn.Linear(in_channels * self.pf * self.ph * self.pw, num_channels)

    def forward(self, x, fhw):
        # shape: x [b, c, f, h, w]
        f, h, w = fhw[0], fhw[1], fhw[2]
        nf, nh, nw = f // self.pf, h // self.ph, w // self.pw

        x_patched = rearrange(x, "b c (nf pf) (nh ph) (nw pw) -> b (nf nh nw) (c pf ph pw)", nf=nf, nh=nh, nw=nw)
        x = self.linear(x_patched)
        return x


class UnPatchify(nn.Module):
    def __init__(self, out_channels, num_channels, patch_size=(1, 1, 1)):
        super().__init__()

        self.patch_size = patch_size
        self.pf, self.ph, self.pw = patch_size
        self.linear = nn.Linear(num_channels, out_channels * self.pf * self.ph * self.pw)
        
    def forward(self, x, fhw):
        # shape of x [bs, seq_len, num_channels]
        f, h, w = fhw[0], fhw[1], fhw[2]
        nf, nh, nw = f // self.pf, h // self.ph, w // self.pw

        x = self.linear(x)
        x = rearrange(x, "b (nf nh nw) (c pf ph pw) -> b c (nf pf) (nh ph) (nw pw)", nf=nf, nh=nh, nw=nw, pf=self.pf, ph=self.ph, pw=self.pw)
        return x


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
    # Saw this one from the code - https://github.com/facebookresearch/DiT/blob/main/models.py#L112
    # and https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py#L14-L54
    def __init__(self, num_channels, mlp_channels):
        super().__init__()
        self.num_channels = num_channels
        
        self.fc1 = nn.Linear(num_channels, mlp_channels)
        self.act1 = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(mlp_channels, num_channels)
    
    def forward(self, x):
        return self.fc2(self.act1(self.fc1(x)))


class DiTBlock(nn.Module):
    def __init__(self, num_channels, num_heads, mlp_ratio):
        super().__init__()

        self.mhsa_norm = nn.LayerNorm(num_channels, elementwise_affine=False)
        self.mhsa = MHSA(num_channels, num_heads)
        self.mlp_norm = nn.LayerNorm(num_channels, elementwise_affine=False)

        mlp_channels = int(num_channels * mlp_ratio)
        self.mlp = Mlp(num_channels, mlp_channels)

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


class FinalLayer(nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        self.norm = nn.LayerNorm(num_channels, elementwise_affine=False)
        self.adaLN_mlp = nn.Linear(num_channels, 2 * num_channels)
        self.linear = nn.Linear(num_channels, num_channels)
    
    def forward(self, x, cond):
        scale, shift = self.adaLN_mlp(cond).chunk(2, dim=-1)
        return self.linear(scale_shift(self.norm(x), scale=scale, shift=shift))


class DiT(nn.Module):
    def __init__(
        self,
        in_channels,
        num_channels,
        depth,
        num_heads,
        patch_size,

        mlp_ratio = 4.0,

        t_dim = 256,
        num_classes = 0
    ):
        super().__init__()

        self.out_channels = self.in_channels = in_channels
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.time_embedder = TimeEmbedder(t_dim, num_channels)
        if self.num_classes > 0:
            self.label_embedder = LabelEmbedder(num_classes, num_channels)
        
        self.patchify = Patchify(in_channels, num_channels, patch_size)
        self.dit_blocks = nn.ModuleList([DiTBlock(num_channels, num_heads, mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(num_channels)
        self.unpatchify = UnPatchify(self.out_channels, num_channels, patch_size=patch_size)

    def forward(self, x, t, conds):
        # shapes
        # x [bs, c, f, h, w]
        # conds will contain condition like t, label, etc.

        b, c, f, h, w = x.shape
        fhw = [f, h, w]

        conditional_embedding = self.time_embedder(t)
        if self.num_classes > 0:
            # Assumption: labels are shifted by 1 such that label 0 represents unconditional
            label = conds.get("label") if "label" in conds else torch.zeros(b, 1)
            label_embedding = self.label_embedder(label).squeeze(1)
            conditional_embedding += label_embedding

        x = self.patchify(x, fhw) # shape [bs, seq_len, num_channels]

        for block in self.dit_blocks:
            x = block(x, conditional_embedding)
        x = self.final_layer(x, conditional_embedding)
        x = self.unpatchify(x, fhw)

        return x


## Some sanity checks
def test_sincos_embedding_1d():
    t = torch.randn(3)
    emb = sincos_embedding_1d(t, 128)
    print(emb.shape)


def test_patchify_unpatchify(bs, c, f, h, w, num_channels):
    patchify = Patchify(c, num_channels, patch_size=(1, 2, 2))
    x = torch.randn(bs, c, f, h, w)
    fhw = x.shape[2:]
    x_patched = patchify(x, fhw)

    unpatchify = UnPatchify(c, num_channels, patch_size=(1, 2, 2))
    x_out = unpatchify(x_patched, fhw)

    assert x.shape == x_out.shape


def test_mhsa(bs, seq_len, num_channels, num_heads):
    mhsa = MHSA(num_channels, num_heads)
    x = torch.randn(bs, seq_len, num_channels)
    output_x = mhsa(x)
    assert output_x.shape == x.shape, f"Output shape {output_x.shape} does not match input shape {x.shape}"


def test_dit_block(bs, seq_len, num_channels, num_heads):
    block = DiTBlock(num_channels, num_heads, mlp_ratio=4.0)
    x = torch.randn(bs, seq_len, num_channels)
    cond = torch.randn(bs, num_channels)

    out = block(x, cond)
    assert out.shape == x.shape, f"Output shape {out.shape} does not match input shape {x.shape}"


def test_dit(bs, c, f, h, w, num_channels, depth, num_heads, num_classes=10, patch_size=(1, 2, 2)):
    dit = DiT(c, num_channels, depth, num_heads, patch_size, num_classes=num_classes)
    x = torch.randn(bs, c, f, h, w)
    conds = {
        "label": torch.zeros(bs, 1, dtype=torch.long)
    }
    t = torch.zeros(bs, 1)
    out = dit(x, t, conds)
    assert out.shape == x.shape, f"Output shape {out.shape} does not match input shape {x.shape}"

if __name__ == '__main__':
    test_sincos_embedding_1d()

    bs, c, f, h, w = 3, 16, 8, 32, 32
    num_channels = 32
    test_patchify_unpatchify(bs, c, f, h, w, num_channels)

    bs, seq_len, num_channels, num_heads = 3, 64, 32, 4

    test_mhsa(bs, seq_len, num_channels, num_heads)
    test_dit_block(bs, seq_len, num_channels, num_heads)
    
    test_dit(bs, c, f, h, w, num_channels, 8, num_heads)