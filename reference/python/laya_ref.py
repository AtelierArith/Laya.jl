"""Helpers that expose laya-mlx internals to Julia for parity testing.

States and questions cross the language boundary as JSON strings so that no
Julia-to-Python container conversion can change their meaning. Arrays are
returned as C-contiguous NumPy arrays; the Julia side reverses their axes.
"""

import json

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from tokenizers import Tokenizer, models, pre_tokenizers

from laya_mlx.agent import Agent, collate_items
from laya_mlx.model import DecisionModel, EncoderConfig, attention_masks

DEVICES = {"gpu": mx.gpu, "cpu": mx.cpu}


def load_agent(path, dtype="float32", device="gpu"):
    return Agent(str(path), dtype=dtype, device=device)


def load_tokenizer(path):
    """Tokenizer only, shaped like an Agent for `tokenize`/`special_tokens` (no weights)."""
    from pathlib import Path
    from types import SimpleNamespace

    from laya_mlx.tokenizer import Tokenizer as LayaTokenizer

    return SimpleNamespace(tok=LayaTokenizer(Path(path) / "tokenizer"), model_id=str(path))


def special_tokens(agent):
    tok = agent.tok
    return {
        name: {"token": getattr(tok, name), "id": getattr(tok, name + "_id")}
        for name in ("cls_token", "sep_token", "pad_token", "mask_token")
    }


def tokenize(agent, text):
    return list(agent.tok(text)["input_ids"])


def prepare(agent, state_json, questions_json):
    items, _ = agent.prepare(json.loads(state_json), json.loads(questions_json))
    return json.dumps(
        [{"ids": list(i["ids"]), "markers": list(i["markers"]), "qtype": i["qtype"]} for i in items]
    )


def collate(agent, items_json, pad_to_multiple=None):
    items = json.loads(items_json)
    batch = collate_items(
        items,
        agent.tok.pad_token_id,
        pad_to_multiple=pad_to_multiple,
        max_length=agent.cfg.get("max_len", 512),
    )
    return {k: np.ascontiguousarray(v) for k, v in batch.items()}


def from_julia(x):
    """A Julia array with reversed axes becomes the row-major NumPy array MLX expects."""
    return np.ascontiguousarray(np.asarray(x).T)


def _np(x):
    return np.ascontiguousarray(np.asarray(x.astype(mx.float32) if x.dtype != mx.bool_ else x))


def trace(agent, batch):
    """Run DecisionModel step by step and return every intermediate activation.

    Mirrors DecisionModel.__call__ exactly; the final logits and action are
    checked against the model's own forward pass.
    """
    model = agent.model
    out = {}
    with mx.stream(agent.device):
        t = {k: mx.array(from_julia(v)) for k, v in batch.items()}
        ids, mask = t["input_ids"], t["attention_mask"]
        enc = model.encoder
        x = enc.embeddings(ids)
        out["embeddings"] = x
        masks = attention_masks(mask, enc.config.local_attention)
        out["mask_full"] = masks["full_attention"]
        out["mask_sliding"] = masks["sliding_attention"]
        for i, layer in enumerate(enc.layers):
            x = layer(x, masks[layer.attention_type])
            out["layer_%d" % i] = x
        h = enc.final_norm(x)
        out["encoder"] = h
        h = h + model.type_emb(t["qtype"])[:, None, :]
        out["typed"] = h
        hmask = mask[:, None, None, :].astype(mx.bool_)
        for i, layer in enumerate(model.head.layers):
            h = layer(h, hmask)
            out["head_%d" % i] = h
        logits, action = model(**t)
        out["logits"] = logits
        out["action"] = action
        mx.eval(out)
    return {k: _np(v) for k, v in out.items()}


def predict(agent, state_json, questions_json):
    result = agent.predict(json.loads(state_json), json.loads(questions_json))
    return json.dumps(result, ensure_ascii=False)


def tiny_checkpoint(path, seed=7):
    """Recreate tests/conftest.py's tiny random checkpoint at `path`."""
    from pathlib import Path

    cfg = {
        "model_type": "modernbert",
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_hidden_layers": 3,
        "num_attention_heads": 1,
        "local_attention": 16,
        "max_position_embeddings": 256,
    }
    agent_cfg = {
        "encoder": "test/tiny",
        "head_layers": 1,
        "max_len": 128,
        "head_max_len": 32,
        "act_costs": {"escalate": 0.5},
        "temperature": [1.3, 1.1, 2.0],
        "temperature_by_options": {"choice:2": 1.7},
    }
    path = Path(path)
    (path / "encoder").mkdir(parents=True)
    (path / "tokenizer").mkdir()
    (path / "encoder/config.json").write_text(json.dumps(cfg))
    (path / "rl_agent_config.json").write_text(json.dumps(agent_cfg))
    vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello"])}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(path / "tokenizer/tokenizer.json"))
    (path / "tokenizer/tokenizer_config.json").write_text(
        json.dumps(
            {
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "sep_token": "[SEP]",
                "mask_token": "[MASK]",
            }
        )
    )
    mx.random.seed(seed)
    model = DecisionModel(EncoderConfig.from_dict(cfg), agent_cfg)
    mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))
    return str(path)
