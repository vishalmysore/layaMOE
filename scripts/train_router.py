"""Train the router that picks an expert head from the encoder states the answers already use.

The prompted router asks the base model an extra question ("What kind of text is this?"), which costs a
second encoder pass per request and misroutes moderation and email cases. This router is a small
classifier over the mean of the frozen encoder's hidden states on the *state* tokens. At inference the
same hidden states feed the expert heads, so routing adds no encoder pass.

Training data: the experts' synthetic data (data/train/<expert>/) plus template texts for the kinds that
stay on the general head (data/train/router/, from gen_router_data.py). Each text is encoded once with
one of its own questions, as at inference. data/eval is never used.

    python scripts/gen_router_data.py
    python scripts/train_router.py                    # -> checkpoints/router/
"""
import argparse, glob, json, os, random, sys, time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from laya_moe.core import encode, load_base, make_item, state_pool_rows  # noqa: E402

DOMAIN_KIND = {"agent-guardrails": "agent_action", "content-moderation": "public_comment",
               "support-tickets": "support_ticket", "delivery-exceptions": "delivery_issue", "email-triage": "work_email"}
KIND_EXPERT = {"agent_action": "safety", "public_comment": "safety", "support_ticket": "customer_ops",
               "delivery_issue": "customer_ops", "work_email": "customer_ops", "it_alert": "general",
               "patient_message": "general", "product_review": "general", "sales_inquiry": "general", "other": "general"}
KINDS = list(KIND_EXPERT)


class RouterHead(nn.Module):
    def __init__(self, dim, n):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(0.1)
        self.out = nn.Linear(dim, n)

    def forward(self, x):
        return self.out(self.drop(self.norm(x)))


def load_rows(root):
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "data", "train", "*", "*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            kind = r.get("kind") or DOMAIN_KIND.get(r.get("domain"))
            if kind:
                rows.append((r["state"], r["questions"], kind))
    return rows


@torch.no_grad()
def featurize(base, rows, bs=16, seed=0):
    rnd = random.Random(seed)
    tok, cfg, enc = base.tok, base.cfg, base.model.encoder
    feats, t0 = [], time.time()
    for i in range(0, len(rows), bs):
        items = []
        for state, qs, _ in rows[i:i + bs]:
            items.append(make_item(tok, cfg, state, qs[rnd.choice(list(qs))]))
        h, *_ = encode(enc, items, tok.pad_token_id)
        feats.append(state_pool_rows(h, items))
        if (i // bs) % 20 == 0:
            print(f"  encoded {i + len(items):5d}/{len(rows)}  ({time.time() - t0:.0f}s)", flush=True)
    return torch.cat(feats)


def main():
    ap = argparse.ArgumentParser()
    root = os.path.join(os.path.dirname(__file__), "..")
    ap.add_argument("--out", default=os.path.join(root, "checkpoints", "router"))
    ap.add_argument("--cache", default=os.path.join(root, "cache", "router_feats.pt"))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    torch.manual_seed(0)

    rows = load_rows(root)
    y = torch.tensor([KINDS.index(k) for *_, k in rows])
    print(f"{len(rows)} training texts: " + ", ".join(f"{k} {int((y == i).sum())}" for i, k in enumerate(KINDS)))
    if os.path.exists(args.cache) and torch.load(args.cache)["n"] == len(rows):
        X = torch.load(args.cache)["X"]
        print("loaded cached features", tuple(X.shape))
    else:
        torch.set_grad_enabled(False)
        base = load_base()
        X = featurize(base, rows)
        os.makedirs(os.path.dirname(args.cache), exist_ok=True)
        torch.save({"X": X, "n": len(rows)}, args.cache)
        torch.set_grad_enabled(True)

    # stratified 85/15 split
    g = torch.Generator().manual_seed(0)
    val = torch.zeros(len(y), dtype=torch.bool)
    for c in range(len(KINDS)):
        idx = (y == c).nonzero().squeeze(1)
        idx = idx[torch.randperm(len(idx), generator=g)]
        val[idx[: max(1, int(0.15 * len(idx)))]] = True
    model = RouterHead(X.size(1), len(KINDS))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    for ep in range(args.epochs):
        model.train()
        loss = nn.functional.cross_entropy(model(X[~val]), y[~val], label_smoothing=0.05)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                acc = (model(X[val]).argmax(1) == y[val]).float().mean()
            print(f"epoch {ep:3d}  loss {loss.item():.3f}  val acc {acc:.1%}")

    model.eval()
    with torch.no_grad():
        pred = model(X[val]).argmax(1)
    exp_acc = sum(KIND_EXPERT[KINDS[a]] == KIND_EXPERT[KINDS[b]] for a, b in zip(pred.tolist(), y[val].tolist())) / int(val.sum())
    print(f"val: kind accuracy {(pred == y[val]).float().mean():.1%}, expert accuracy {exp_acc:.1%}")

    from safetensors.torch import save_file
    os.makedirs(args.out, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, os.path.join(args.out, "router.safetensors"))
    json.dump({"kinds": KINDS, "kind_to_expert": KIND_EXPERT, "dim": X.size(1), "threshold": args.threshold,
               "pooling": "mean of encoder states over state tokens, averaged over question rows",
               "train_texts": len(rows), "val_kind_acc": float((pred == y[val]).float().mean()), "val_expert_acc": exp_acc},
              open(os.path.join(args.out, "router.json"), "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
