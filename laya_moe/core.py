"""Shared pieces for Laya MoE: loading the base checkpoint, splitting it into a frozen encoder and a
trainable decision head ("expert"), and turning (state, question) pairs into model inputs.

The Laya forward pass is  encoder(ids) -> + type_emb(qtype) -> 2 transformer layers -> scorer at the
option [MASK] markers. Everything after the encoder is small (~27M params), so an expert is a copy of
exactly those modules. The encoder (~395M params) is shared by every expert and never trained.
"""
import copy
import json
import os

import torch
import torch.nn as nn

from laya.agent import Agent
from laya.common import QTYPES, build_sequence, render_options

BASE_MODEL = "convaiinnovations/laya-typed-decisions"
HEAD_PREFIXES = ("type_emb.", "head.", "scorer.", "act_head.")


def load_base(model_id=BASE_MODEL):
    """Load the base Laya checkpoint on CPU in eval mode."""
    agent = Agent(model_id, device="cpu")
    agent.model.eval()
    return agent


def to_internal(qdef):
    return Agent._to_internal(qdef)


def make_item(tok, cfg, state, qdef):
    """Token ids + marker positions for one (state, question) pair, exactly as laya builds them."""
    q = to_internal(qdef)
    ids, markers = build_sequence(tok, state, q, cfg.get("max_len", 512), cfg.get("head_max_len", 192))
    if len(markers) != len(render_options(q)):
        raise ValueError("options exceed head_max_len")
    return {"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]}


def label_index(qdef, expected):
    """Index of the correct option for a question spec (choice: option name, score: level, noul: bool)."""
    t = qdef["type"]
    if t == "choice":
        crit = qdef["criteria"]
        keys = list(crit.keys()) if isinstance(crit, dict) else list(crit)
        return keys.index(expected)
    if t == "score":
        return int(expected)
    return 1 if expected else 0


class ExpertHead(nn.Module):
    """The part of Laya's DecisionModel that runs after the encoder."""

    def __init__(self, base_model):
        super().__init__()
        self.type_emb = copy.deepcopy(base_model.type_emb)
        self.head = copy.deepcopy(base_model.head)
        self.scorer = copy.deepcopy(base_model.scorer)
        self.act_head = copy.deepcopy(base_model.act_head)

    def forward(self, h, attention_mask, marker_pos, marker_mask, qtype):
        h = h + self.type_emb(qtype)[:, None, :]
        pad = ~attention_mask.bool()
        for layer in self.head.layers:
            h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        return logits.masked_fill(~marker_mask, -1e4)


def save_expert(path, head, meta):
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True)
    save_file({k: v.contiguous() for k, v in head.state_dict().items()}, os.path.join(path, "expert.safetensors"))
    with open(os.path.join(path, "expert.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)


def load_expert(path, base_model):
    from safetensors.torch import load_file
    head = ExpertHead(base_model)
    head.load_state_dict(load_file(os.path.join(path, "expert.safetensors")), strict=True)
    head.eval()
    meta = json.load(open(os.path.join(path, "expert.json"), encoding="utf-8"))
    return head, meta


@torch.no_grad()
def encode(encoder, items, pad_id):
    """Run the frozen encoder on a list of items; returns padded tensors for the head."""
    L = max(len(it["ids"]) for it in items)
    K = max(len(it["markers"]) for it in items)
    n = len(items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, K), dtype=torch.long)
    mmask = torch.zeros((n, K), dtype=torch.bool)
    for i, it in enumerate(items):
        ids[i, :len(it["ids"])] = torch.tensor(it["ids"])
        att[i, :len(it["ids"])] = 1
        mpos[i, :len(it["markers"])] = torch.tensor(it["markers"])
        mmask[i, :len(it["markers"])] = True
    h = encoder(input_ids=ids, attention_mask=att).last_hidden_state
    return h, att, mpos, mmask, torch.tensor([it["qtype"] for it in items])
