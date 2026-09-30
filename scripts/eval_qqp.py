"""Duplicate-question detection on GLUE QQP: original Laya vs layaMOE (and optionally a QQP expert head).

Eval slice: a balanced, stratified sample of 1,500 pairs from the QQP validation split (750 duplicate,
750 not duplicate, random_state=42), the setup used in public GLiNER2.5-Decide / Jev comparisons.
Each pair is one noul question: P(the two questions ask the same thing).

    python scripts/eval_qqp.py                                   # zero-shot: original + MoE (trained router)
    python scripts/eval_qqp.py --pairs-expert checkpoints/pairs  # also score a head trained on QQP
                                                                 # (gen_qqp_data.py -> cache_features -> train_expert)

Needs the QQP validation parquet (downloaded automatically from the Hugging Face hub).
"""
import argparse, json, os, sys, time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from laya_moe.core import encode, load_expert, make_item  # noqa: E402
from laya_moe.moe import MoEAgent  # noqa: E402

QUESTION = {"type": "noul", "instructions": "Do question1 and question2 ask the same thing?"}


def load_slice(n_per_class=750, seed=42):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("nyu-mll/glue", "qqp/validation-00000-of-00001.parquet", repo_type="dataset")
    d = pd.read_parquet(p)
    return d.groupby("label", group_keys=False).sample(n=n_per_class, random_state=seed).reset_index(drop=True)


def metrics(p_yes, y, thr=0.5):
    pred = p_yes >= thr
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": (tp + tn) / len(y), "f1": f1, "precision": prec, "recall": rec,
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}}


def main():
    root = os.path.join(os.path.dirname(__file__), "..")
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", nargs="+", default=[os.path.join(root, "checkpoints", e) for e in ("safety", "customer_ops")])
    ap.add_argument("--router", default=os.path.join(root, "checkpoints", "router"))
    ap.add_argument("--pairs-expert", default=None, help="a head trained on QQP (checkpoints/pairs)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--serial", type=int, default=100, help="pairs timed one at a time for serial latency")
    ap.add_argument("--out", default=os.path.join(root, "results", "qqp_eval.json"))
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    df = load_slice()
    y = df.label.values.astype(bool)
    states = [{"question1": a, "question2": b} for a, b in zip(df.question1, df.question2)]
    moe = MoEAgent(experts=args.experts, router=args.router)
    if args.pairs_expert:
        head, meta = load_expert(args.pairs_expert, moe.base.model)
        moe.experts["pairs"] = (head, meta)
    heads = ["general"] + list(moe.experts)

    p = {h: np.zeros(len(df)) for h in heads}
    routed, t0 = [], time.perf_counter()
    for i in range(0, len(df), args.bs):
        chunk = states[i:i + args.bs]
        qs = {f"r{j}": QUESTION for j in range(len(chunk))}
        # one encoder pass for the whole batch; each row is (pair, question)
        items = [make_item(moe.tok, moe.cfg, s, QUESTION) for s in chunk]
        enc = encode(moe.base.model.encoder, items, moe.tok.pad_token_id)
        for h in heads:
            ans = moe._answers(h, qs, list(qs), items, enc)
            p[h][i:i + len(chunk)] = [ans[f"r{j}"]["noul"] for j in range(len(chunk))]
        for j in range(len(chunk)):
            routed.append(moe.route_states(enc[0][j:j + 1], [items[j]])[0])
        if (i // args.bs) % 10 == 0:
            print(f"  {i + len(chunk)}/{len(df)}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    batch_ms = 1000 * (time.perf_counter() - t0) / len(df)

    # the MoE answer is whichever head the trained router picked (pairs expert excluded: it is a separate test)
    p_moe = np.array([p[r if r in p else "general"][k] for k, r in enumerate(routed)])

    # serial latency: one pair per request, as an application would call it
    t0 = time.perf_counter()
    for s in states[:args.serial]:
        moe.system_one(s, {"dup": QUESTION})
    serial_ms = 1000 * (time.perf_counter() - t0) / args.serial

    res = {"slice": "QQP validation, 750 duplicate + 750 not, random_state=42", "n": len(df),
           "original": metrics(p["general"], y), "moe_trained_router": metrics(p_moe, y),
           "routing": {h: routed.count(h) for h in set(routed)},
           "latency_ms": {"serial_per_pair_moe": serial_ms, "batched_per_pair_all_heads": batch_ms, "bs": args.bs,
                          "threads": torch.get_num_threads()}}
    for h in heads:
        if h != "general":
            res[f"head_{h}"] = metrics(p[h], y)
    # best threshold on the eval slice itself is optimistic; reported only to show calibration offset
    res["original_best_threshold"] = max(((t, metrics(p["general"], y, t)["accuracy"]) for t in np.arange(0.05, 0.96, 0.05)),
                                         key=lambda x: x[1])
    print(json.dumps({k: v for k, v in res.items()}, indent=1, default=float))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({**res, "p_yes": {h: v.round(4).tolist() for h, v in p.items()}, "routed": routed,
               "labels": y.astype(int).tolist(), "idx": df.idx.tolist()}, open(args.out, "w"), indent=1, default=float)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
