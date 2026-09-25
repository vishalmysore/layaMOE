"""Run the frozen Laya encoder once over every training item and cache its hidden states.

Training an expert then only needs the small head, which is cheap on a CPU.

    python scripts/cache_features.py --expert safety
    python scripts/cache_features.py --expert customer_ops

Writes cache/<expert>.pt: a list of items with fp16 hidden states, marker positions, qtype,
label, split ("train" / "val", split by state so one state's questions never straddle both), and
metadata (domain, question name, number of options).
"""
import argparse, glob, hashlib, json, os, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from laya_moe.core import encode, label_index, load_base, make_item  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def split_of(state, val_frac):
    h = int(hashlib.sha1(json.dumps(state, sort_keys=True).encode()).hexdigest(), 16)
    return "val" if (h % 1000) < val_frac * 1000 else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    agent = load_base()
    tok, cfg, enc = agent.tok, agent.cfg, agent.model.encoder

    items = []
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "train", args.expert, "*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            sp = split_of(row["state"], args.val_frac)
            for qname, qdef in row["questions"].items():
                it = make_item(tok, cfg, row["state"], qdef)
                it.update(label=label_index(qdef, row["expected"][qname]), split=sp, domain=row["domain"],
                          question=qname, k=len(it["markers"]))
                items.append(it)
    if not items:
        sys.exit("no training data for expert %r" % args.expert)
    print(f"{len(items)} items ({sum(i['split'] == 'val' for i in items)} val)")

    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    t0 = time.time()
    for bi in range(0, len(order), args.batch):
        batch = [items[i] for i in order[bi:bi + args.batch]]
        h = encode(enc, batch, tok.pad_token_id)[0]
        for j, it in enumerate(batch):
            it["h"] = h[j, :len(it["ids"])].to(torch.float16).clone()
        if (bi // args.batch) % 20 == 0:
            done = bi + len(batch)
            print(f"  {done}/{len(items)}  {time.time() - t0:.0f}s", flush=True)

    os.makedirs(os.path.join(ROOT, "cache"), exist_ok=True)
    out = os.path.join(ROOT, "cache", args.expert + ".pt")
    torch.save(items, out)
    print(f"wrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
