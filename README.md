Start diffusion model training on cifar10 dataset

```
torchrun --nproc_per_node=1 main.py --config configs/cifar10.yaml
```

Install command for FA3
```sh
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291 --extra-index-url https://download.pytorch.org/whl/cu128
```