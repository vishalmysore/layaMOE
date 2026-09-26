#!/usr/bin/env python3
"""Export Laya MoE for the browser: one shared encoder graph + one small graph per head.

    python scripts/export_web.py                         # all experts in checkpoints/ + the general head
    python scripts/export_web.py --skip-encoder          # re-export heads only (encoder unchanged)

Unlike layaForWeb (one graph for encoder + head), the graphs are split at the encoder output:

    encoder.onnx   input_ids, attention_mask                            -> hidden [B, L, 1024]
    head_X.onnx    hidden, attention_mask, marker_pos, marker_mask, qtype -> logits [B, K]

so the page downloads the ~400 MB encoder once, runs it once per request (router question and the
user's questions in one batch), and then runs whichever ~27 MB head the router picked, or several
heads on the same hidden states to compare them.

Quantization is weight-only, like layaForWeb's default q8e8 build: int8 block-128 MatMulNBits for
matrix weights and int8 per-row embeddings. Every graph is checked against PyTorch before packaging.

Output: build/web/ -> upload to the Hugging Face web repo (scripts/upload_hf.py --web).
"""
import argparse, gc, glob, hashlib, json, os, shutil, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from laya_moe.core import BASE_MODEL, ExpertHead, encode, load_base, load_expert, make_item  # noqa: E402
from laya_moe.moe import ROUTER_EXPERTS, ROUTER_OPTIONS, ROUTER_QUESTION  # noqa: E402

CHUNK = 24 * 1024 * 1024


def log(m):
    print(f"[export] {m}", flush=True)


# ------------------------------------------------------------------------------------------ export
def sample_items(agent):
    state = {"ticket": {"subject": "App crashes on launch", "text": "Since the last update, the app closes as soon as I open it."}}
    qs = [
        {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"bug": "Something is broken", "how_to": "A usage question", "sales": "Pricing or plans"}},
        {"type": "score", "instructions": "How urgent is this?", "criteria": ["Can wait", "This week", "Today", "Right now"]},
        {"type": "noul", "instructions": "The customer sounds angry"},
    ]
    return [make_item(agent.tok, agent.cfg, state, q) for q in qs]


def export_encoder(agent, out):
    from torch.export import Dim

    class Enc(torch.nn.Module):
        def __init__(self, e):
            super().__init__()
            self.e = e

        def forward(self, input_ids, attention_mask):
            return self.e(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    items = sample_items(agent)
    ids = torch.full((3, max(len(i["ids"]) for i in items)), agent.tok.pad_token_id)
    att = torch.zeros_like(ids)
    for r, it in enumerate(items):
        ids[r, :len(it["ids"])] = torch.tensor(it["ids"])
        att[r, :len(it["ids"])] = 1
    batch, seq = Dim("batch", min=1, max=64), Dim("seq", min=8, max=int(agent.cfg.get("max_len", 512)))
    prog = torch.onnx.export(Enc(agent.model.encoder).eval(), (ids, att), dynamo=True,
                             dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
                             input_names=["input_ids", "attention_mask"], output_names=["hidden"], opset_version=18, optimize=True)
    raw = out / "encoder_fp32.onnx"
    prog.save(str(raw), external_data=True)
    del prog
    gc.collect()
    return raw


def export_head(head, agent, out, name):
    from torch.export import Dim
    items = sample_items(agent)
    with torch.no_grad():
        h, att, mpos, mmask, qt = encode(agent.model.encoder, items, agent.tok.pad_token_id)
    batch, seq, kk = Dim("batch", min=1, max=64), Dim("seq", min=8, max=int(agent.cfg.get("max_len", 512))), Dim("k", min=2, max=255)
    dyn = {"hidden": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq},
           "marker_pos": {0: batch, 1: kk}, "marker_mask": {0: batch, 1: kk}, "qtype": {0: batch}}

    class Head(torch.nn.Module):
        def __init__(self, hd):
            super().__init__()
            self.hd = hd

        def forward(self, hidden, attention_mask, marker_pos, marker_mask, qtype):
            return self.hd(hidden, attention_mask, marker_pos, marker_mask, qtype)

    prog = torch.onnx.export(Head(head.eval()).eval(), (h, att, mpos, mmask, qt), dynamo=True, dynamic_shapes=dyn,
                             input_names=list(dyn), output_names=["logits"], opset_version=18, optimize=True)
    raw = out / f"head_{name}_fp32.onnx"
    prog.save(str(raw), external_data=True)
    del prog
    gc.collect()
    return raw


# ---------------------------------------------------------------------------------------- quantize
def _embedding_to_int8(m):
    from onnx import numpy_helper, helper, TensorProto
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    new_nodes, drop, added = [], set(), []
    for node in g.node:
        if node.op_type == "Gather" and node.input[0] in inits:
            t = inits[node.input[0]]
            if t.data_type == TensorProto.FLOAT and len(t.dims) == 2 and t.dims[0] > 1000:
                w = numpy_helper.to_array(t).astype(np.float32)
                scale = np.maximum(np.abs(w).max(axis=1, keepdims=True) / 127.0, 1e-8).astype(np.float32)
                q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
                n = t.name
                added += [numpy_helper.from_array(q, n + "_q8"), numpy_helper.from_array(scale.reshape(-1), n + "_scale")]
                new_nodes.append(helper.make_node("DequantizeLinear", [n + "_q8", n + "_scale"], [n + "_dq"], axis=0, name=n + "_DQ"))
                drop.add(n)
                node.input[0] = n + "_dq"
    keep = [i for i in g.initializer if i.name not in drop]
    del g.initializer[:]
    g.initializer.extend(keep + added)
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(new_nodes + nodes)


def quantize_q8(raw, dst):
    """int8 block-128 MatMulNBits weights (+ int8 embeddings if the graph has an embedding table)."""
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig
    m = onnx.load(str(raw), load_external_data=True)
    del m.graph.value_info[:]  # stale shape annotations from the exporter trip the quantizer
    q = MatMulNBitsQuantizer(m, algo_config=DefaultWeightOnlyQuantConfig(block_size=128, is_symmetric=True, bits=8))
    q.process()
    mq = q.model.model
    _embedding_to_int8(mq)
    onnx.save(mq, str(dst), save_as_external_data=True, all_tensors_to_one_file=True, location=dst.name + ".data", size_threshold=1024)
    del q, m, mq
    gc.collect()
    log(f"{dst.name}: {(dst.parent / (dst.name + '.data')).stat().st_size / 1e6:.0f} MB")


# ------------------------------------------------------------------------------------------ verify
def verify(agent, heads, onnx_dir):
    """Run a few real questions through PyTorch and through the quantized graphs; report the drift."""
    import onnxruntime as ort
    cases = [
        ("Agent plan: run `DROP TABLE invoices` on the production database. No backup exists.",
         {"type": "noul", "instructions": "The action is destructive and cannot be undone"}),
        ("Agent plan: run `DROP TABLE invoices` on the production database. No backup exists.",
         {"type": "score", "instructions": "How risky is this action?", "criteria": ["Low", "Medium", "High", "Critical"]}),
        ("Parcel is stuck at customs; the customer is furious and wants a refund.",
         {"type": "choice", "instructions": "What should support do?", "criteria": {"reship": "Send a replacement", "notify": "Tell the customer",
                                                                                    "fix_address": "Fix the address", "wait": "Wait"}}),
        ("Parcel is stuck at customs; the customer is furious and wants a refund.", ROUTER_QUESTION),
    ]
    items = [make_item(agent.tok, agent.cfg, s, q) for s, q in cases]
    with torch.no_grad():
        h, att, mpos, mmask, qt = encode(agent.model.encoder, items, agent.tok.pad_token_id)
    so = ort.SessionOptions()
    enc = ort.InferenceSession(str(onnx_dir / "encoder.onnx"), so, providers=["CPUExecutionProvider"])
    ho = enc.run(None, {"input_ids": _ids(items, agent), "attention_mask": att.numpy().astype(np.int64)})[0]
    log(f"encoder hidden max |diff| vs PyTorch: {np.abs(ho - h.numpy()).max():.4f}")
    worst = 0.0
    for name, head in heads.items():
        with torch.no_grad():
            ref = torch.softmax(head(h, att, mpos, mmask, qt), -1).numpy()
        sess = ort.InferenceSession(str(onnx_dir / f"head_{name}.onnx"), so, providers=["CPUExecutionProvider"])
        lg = sess.run(None, {"hidden": ho.astype(np.float32), "attention_mask": att.numpy().astype(np.int64), "marker_pos": mpos.numpy().astype(np.int64),
                             "marker_mask": mmask.numpy(), "qtype": qt.numpy().astype(np.int64)})[0]
        p = np.exp(lg - lg.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        d = max(np.abs(p[r, :len(it["markers"])] - ref[r, :len(it["markers"])]).max() for r, it in enumerate(items))
        same = all(p[r, :len(it["markers"])].argmax() == ref[r, :len(it["markers"])].argmax() for r, it in enumerate(items))
        log(f"head {name}: max |dp| {d:.4f}, same top answer on all {len(items)}: {same}")
        worst = max(worst, d)
    if worst > 0.15:
        sys.exit(f"quantized graphs drift too far from PyTorch (max |dp| {worst:.3f})")


def _ids(items, agent):
    L = max(len(i["ids"]) for i in items)
    ids = np.full((len(items), L), agent.tok.pad_token_id, dtype=np.int64)
    for r, it in enumerate(items):
        ids[r, :len(it["ids"])] = it["ids"]
    return ids


# ------------------------------------------------------------------------------------------ package
def split(data_path, dst_dir):
    parts, h, size = [], hashlib.sha256(), data_path.stat().st_size
    with open(data_path, "rb") as f:
        i = 0
        while buf := f.read(CHUNK):
            h.update(buf)
            name = f"{data_path.name}.part{i:03d}"
            (dst_dir / name).write_bytes(buf)
            parts.append(name)
            i += 1
    return {"name": data_path.name, "size": size, "sha256": h.hexdigest(), "parts": parts}


def package(agent, metas, onnx_dir, web_dir):
    if web_dir.exists():
        shutil.rmtree(web_dir)
    web_dir.mkdir(parents=True)
    snap = Path(agent_dir(agent))
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(snap / "tokenizer" / f, web_dir / f)
    shutil.copy(snap / "rl_agent_config.json", web_dir / "rl_agent_config.json")
    manifest = {"version": 1, "base_model": BASE_MODEL, "chunk_bytes": CHUNK,
                "router": {"question": ROUTER_QUESTION["instructions"], "options": ROUTER_OPTIONS, "experts": ROUTER_EXPERTS, "threshold": 0.3},
                "encoder": {"onnx": "encoder.onnx", "data": None}, "heads": {}}
    shutil.copy(onnx_dir / "encoder.onnx", web_dir / "encoder.onnx")
    manifest["encoder"]["data"] = split(onnx_dir / "encoder.onnx.data", web_dir)
    for name, meta in metas.items():
        shutil.copy(onnx_dir / f"head_{name}.onnx", web_dir / f"head_{name}.onnx")
        manifest["heads"][name] = {"onnx": f"head_{name}.onnx", "data": split(onnx_dir / f"head_{name}.onnx.data", web_dir), **meta}
    (web_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for f in ("LICENSE", "NOTICE.md"):
        shutil.copy(ROOT / f, web_dir / f)
    (web_dir / "README.md").write_text(MODEL_CARD)
    total = sum(p.stat().st_size for p in web_dir.iterdir())
    log(f"web model folder ready: {web_dir} ({total / 1e6:.0f} MB)")


def agent_dir(agent):
    from huggingface_hub import snapshot_download
    return snapshot_download(BASE_MODEL, allow_patterns=["tokenizer/*", "rl_agent_config.json"])


MODEL_CARD = f"""---
license: apache-2.0
base_model: {BASE_MODEL}
library_name: onnx
tags: [onnx, onnxruntime-web, browser, decision-model, mixture-of-experts, quantized]
---
# Laya MoE for the browser

ONNX Runtime Web files for [layaMOE](https://github.com/vishalmysore/layaMOE): the shared encoder of
[{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) plus one small decision head per expert
(`general` is the original head; the others were fine-tuned on synthetic domain data).

- `encoder.onnx` + parts: ModernBERT-large encoder, int8 weights (~400 MB, downloaded once)
- `head_<name>.onnx` + parts: 2-layer decision head per expert, int8 weights
- `manifest.json`: file list with SHA-256, router question, per-head calibration temperatures

Modified derivative of {BASE_MODEL} (Apache-2.0, Copyright ConvAI Innovations); unofficial and not affiliated
with ConvAI Innovations. Changes: split into encoder and head graphs, heads fine-tuned, weights quantized
(weight-only int8), files split into parts. See `LICENSE` and `NOTICE.md`.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", nargs="*", default=sorted(glob.glob(str(ROOT / "checkpoints" / "*"))))
    ap.add_argument("--out", default=str(ROOT / "build"))
    ap.add_argument("--skip-encoder", action="store_true")
    args = ap.parse_args()
    torch.backends.mha.set_fastpath_enabled(False)  # keep nn.TransformerEncoderLayer exportable
    out = Path(args.out)
    onnx_dir = out / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    agent = load_base()
    agent.model.float().eval()
    heads = {"general": ExpertHead(agent.model).eval()}
    metas = {"general": {"label": "General (original laya-typed-decisions head)", "domains": [],
                         "temperature": agent.cfg.get("temperature", [1, 1, 1]),
                         "temperature_by_options": agent.cfg.get("temperature_by_options", {})}}
    for path in args.experts:
        head, meta = load_expert(path, agent.model)
        heads[meta["expert"]] = head
        metas[meta["expert"]] = {"label": meta["expert"], "domains": meta["domains"], "temperature": meta["temperature"],
                                 "temperature_by_options": meta["temperature_by_options"], "train": meta.get("train", {})}
    log(f"heads: {list(heads)}")

    if not (args.skip_encoder and (onnx_dir / "encoder.onnx").exists()):
        log("exporting encoder...")
        raw = export_encoder(agent, onnx_dir)
        quantize_q8(raw, onnx_dir / "encoder.onnx")
    for name, head in heads.items():
        log(f"exporting head {name}...")
        raw = export_head(head, agent, onnx_dir, name)
        quantize_q8(raw, onnx_dir / f"head_{name}.onnx")

    verify(agent, heads, onnx_dir)
    package(agent, metas, onnx_dir, out / "web")


if __name__ == "__main__":
    main()
