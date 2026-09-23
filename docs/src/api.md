# API reference

```@docs
Laya
```

## Loading and prediction

```@docs
load
predict
system_one
Agent
Laya.resolve_model
Laya.hub_download
```

## Backends

```@docs
Backend
CPUBackend
AccelerateBackend
Laya.load_backend_model
```

## Model

```@docs
DecisionModel
load_model
load_safetensors
EncoderConfig
Tokenizer
Laya.prepare
Laya.collate
```

## Array-generic building blocks

The model code is written against `AbstractArray`; device backends move the weights and
specialize these by array type.

```@docs
Laya.adapt_arrays
Laya.on_device_of
Laya.to_host
Laya.gather_columns
Laya.release!
Laya.residual_norm
Laya.qkv_attention
Laya.AttentionMask
Laya.attention
Laya.rope
```
