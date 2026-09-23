# Checkpoints

| repository | encoder | parameters |
|---|---|---:|
| [`aac6fef/laya-mlx`](https://huggingface.co/aac6fef/laya-mlx) | ModernBERT-large | 421M |
| [`aac6fef/laya-multilingual-mlx`](https://huggingface.co/aac6fef/laya-multilingual-mlx) | mmBERT-base | 322M |

These MLX-converted checkpoints (weights stored in `Float16`) are the ones tested. The compute
type is chosen with `dtype` when loading.

[`load`](@ref) accepts either of the following (see [`Laya.resolve_model`](@ref)):

- **A local directory** containing `model.safetensors`, `rl_agent_config.json`,
  `encoder/config.json` and `tokenizer/`.
- **A Hugging Face repository id**, looked up in this order:
  1. the Hugging Face cache (`HF_HUB_CACHE`, `HF_HOME/hub` or `~/.cache/huggingface/hub`),
     shared with Python;
  2. Laya's Scratch.jl space;
  3. otherwise the repository is downloaded into that scratch space ([`Laya.hub_download`](@ref)).

  Downloading respects `HF_HUB_OFFLINE`, `HF_TOKEN` and `HF_ENDPOINT`.

The `Float32` model of the 421M checkpoint needs about 3 GiB of memory.
