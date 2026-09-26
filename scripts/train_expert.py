"""Train one domain expert: a copy of the laya-typed-decisions head, fine-tuned on cached encoder
features (see cache_features.py). The encoder stays frozen and shared.

    python scripts/train_expert.py --expert safety
    python scripts/train_expert.py --expert customer_ops --epochs 6 --lr 6e-4

Loss: cross-entropy over the options, plus the ranked probability score on score questions (it
penalises mass far from the right level, which targets the base model's habit of answering the
middle of every scale). The best epoch on the validation split is kept, then a temperature per
(question type, option count) bucket is fitted on validation, the same way Laya calibrates.

Writes checkpoints/<expert>/expert.safetensors + expert.json.
"""
import argparse, math, os, random, sys, time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from laya.common import QTYPE_NAMES, temp_bucket  # noqa: E402
from laya_moe.core import ExpertHead, load_base, save_expert  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def batches(items, bs, shuffle):
    order = sorted(range(len(items)), key=lambda i: items[i]["h"].shape[0])
    chunks = [order[i:i + bs] for i in range(0, len(order), bs)]
    if shuffle:
        random.shuffle(chunks)
    for ch in chunks:
        its = [items[i] for i in ch]
        L = max(it["h"].shape[0] for it in its)
        K = max(it["k"] for it in its)
        n, d = len(its), its[0]["h"].shape[1]
        h = torch.zeros((n, L, d))
        att = torch.zeros((n, L), dtype=torch.long)
        mpos = torch.zeros((n, K), dtype=torch.long)
        mmask = torch.zeros((n, K), dtype=torch.bool)
        for j, it in enumerate(its):
            l = it["h"].shape[0]
            h[j, :l] = it["h"].float()
            att[j, :l] = 1
            mpos[j, :it["k"]] = torch.tensor(it["markers"])
            mmask[j, :it["k"]] = True
        yield its, h, att, mpos, mmask, torch.tensor([it["qtype"] for it in its]), torch.tensor([it["label"] for it in its])


def loss_fn(logits, labels, qtype, mmask, rps_w):
    ce = F.cross_entropy(logits, labels)
    is_score = qtype == 1
    if rps_w and is_score.any():
        p = torch.softmax(logits[is_score], -1)
        t = F.one_hot(labels[is_score], logits.size(-1)).float()
        k = mmask[is_score].sum(-1).clamp(min=2).float()
        rps = ((torch.cumsum(p, -1) - torch.cumsum(t, -1)) ** 2 * mmask[is_score]).sum(-1) / (k - 1)
        ce = ce + rps_w * rps.mean()
    return ce


@torch.no_grad()
def evaluate(head, items, bs=64):
    head.eval()
    out = []
    for its, h, att, mpos, mmask, qt, lab in batches(items, bs, False):
        logits = head(h, att, mpos, mmask, qt)
        for j, it in enumerate(its):
            out.append((it, logits[j, :it["k"]].clone()))
    stats = {}
    for it, lg in out:
        key = (it["domain"], QTYPE_NAMES[it["qtype"]])
        s = stats.setdefault(key, [0, 0])
        s[0] += 1
        s[1] += int(lg.argmax().item() == it["label"])
    acc = sum(v[1] for v in stats.values()) / max(1, sum(v[0] for v in stats.values()))
    return acc, stats, out


def fit_temperatures(out):
    """Grid-search one temperature per laya bucket (and per qtype) minimising validation NLL."""
    grid = [0.3 + 0.05 * i for i in range(55)]  # 0.30 .. 3.0

    def best_t(pairs):
        best = (1.0, float("inf"))
        for T in grid:
            nll = sum(-F.log_softmax(lg / T, -1)[it["label"]].item() for it, lg in pairs) / len(pairs)
            if nll < best[1]:
                best = (T, nll)
        return best[0]

    by_bucket, by_type = {}, {}
    for it, lg in out:
        by_bucket.setdefault(temp_bucket(it["qtype"], it["k"]), []).append((it, lg))
        by_type.setdefault(it["qtype"], []).append((it, lg))
    per_type = [round(best_t(by_type[q]), 3) if q in by_type else 1.0 for q in range(3)]
    per_bucket = {b: round(best_t(p), 3) for b, p in by_bucket.items() if len(p) >= 20}
    return per_type, per_bucket


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--rps", type=float, default=1.0, help="weight of the ranked probability score on score questions")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    items = torch.load(os.path.join(ROOT, "cache", args.expert + ".pt"), weights_only=False)
    train = [i for i in items if i["split"] == "train"]
    val = [i for i in items if i["split"] == "val"]
    print(f"expert {args.expert}: {len(train)} train items, {len(val)} val items", flush=True)

    base = load_base()
    head = ExpertHead(base.model)
    del base

    acc0, stats0, _ = evaluate(head, val)
    print(f"epoch 0 (base head)  val acc {acc0:.3f}", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train) / args.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, (steps - s) / steps))
    best = (acc0, {k: v.clone() for k, v in head.state_dict().items()}, 0)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        head.train()
        tot, n, nb = 0.0, 0, math.ceil(len(train) / args.bs)
        for bi, (its, h, att, mpos, mmask, qt, lab) in enumerate(batches(train, args.bs, True)):
            logits = head(h, att, mpos, mmask, qt)
            loss = loss_fn(logits, lab, qt, mmask, args.rps)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(its)
            n += len(its)
            if bi % 20 == 0:
                print(f"  epoch {ep} batch {bi}/{nb}  loss {tot / n:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        acc, _, _ = evaluate(head, val)
        print(f"epoch {ep}  train loss {tot / n:.4f}  val acc {acc:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if acc > best[0]:
            best = (acc, {k: v.clone() for k, v in head.state_dict().items()}, ep)

    head.load_state_dict(best[1])
    acc, stats, out = evaluate(head, val)
    per_type, per_bucket = fit_temperatures(out)
    print(f"\nkept epoch {best[2]}  val acc {acc:.3f}  (base head {acc0:.3f})")
    print(f"{'domain':22} {'type':7} {'n':>4} {'base':>6} {'expert':>7}")
    for key in sorted(stats):
        n0, c0 = stats0[key]
        n1, c1 = stats[key]
        print(f"{key[0]:22} {key[1]:7} {n1:4d} {c0 / n0:6.3f} {c1 / n1:7.3f}")
    print("temperature", per_type, per_bucket)

    domains = sorted({i["domain"] for i in items})
    save_expert(os.path.join(ROOT, "checkpoints", args.expert), head, {
        "expert": args.expert, "base_model": "convaiinnovations/laya-typed-decisions", "domains": domains,
        "temperature": per_type, "temperature_by_options": per_bucket,
        "train": {"items": len(train), "val_items": len(val), "epochs_run": args.epochs, "kept_epoch": best[2],
                  "lr": args.lr, "bs": args.bs, "rps_weight": args.rps, "seed": args.seed,
                  "val_acc": round(acc, 4), "val_acc_base_head": round(acc0, 4)},
    })
    print("saved checkpoints/%s" % args.expert)


if __name__ == "__main__":
    main()
