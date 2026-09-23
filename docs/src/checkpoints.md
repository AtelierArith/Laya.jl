# Checkpoints

| repository | encoder | parameters |
|---|---|---:|
| [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) (the default of `load()`) | ModernBERT-large | 421M |
| [`aac6fef/laya-mlx`](https://huggingface.co/aac6fef/laya-mlx) (MLX conversion of the above) | ModernBERT-large | 421M |
| [`aac6fef/laya-multilingual-mlx`](https://huggingface.co/aac6fef/laya-multilingual-mlx) | mmBERT-base | 322M |

- **Original checkpoints**: the upstream repository `convaiinnovations/laya` also holds the
  multilingual and typed-decisions checkpoints in subfolders:
  `Laya.load("convaiinnovations/laya"; subfolder="multilingual")`.
- **Downloads**: only the files of the requested checkpoint are downloaded, as upstream does.
- **Tested**: the original and the MLX-converted English checkpoints give the same answers.
  The MLX-converted checkpoints are the ones compared against the Python reference.

The weights are stored in 16 bits; the compute type is chosen with `dtype` when loading.

[`load`](@ref) accepts either of the following (see [`Laya.resolve_model`](@ref)):

- **A local directory** containing `model.safetensors`, `rl_agent_config.json`,
  `encoder/config.json` and `tokenizer/`.
- **A Hugging Face repository id**, looked up in this order:
  1. the Hugging Face cache (`HF_HUB_CACHE`, `HF_HOME/hub` or `~/.cache/huggingface/hub`),
     shared with Python;
  2. Laya's Scratch.jl space;
  3. otherwise the checkpoint's files (not the whole repository) are downloaded into that
     scratch space ([`Laya.hub_download`](@ref)).

  Downloading respects `HF_HUB_OFFLINE`, `HF_TOKEN` and `HF_ENDPOINT`.

The `Float32` model of the 421M checkpoint needs about 3 GiB of memory.
