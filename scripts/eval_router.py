"""Compare original Laya, the MoE with the prompted router, and the MoE with the trained router on data/eval.

For every case it records accuracy, which head answered, and wall-clock time per request:
  original  - the base laya-typed-decisions head (one encoder pass)
  prompted  - MoE, router = an extra base-model question (two encoder passes)
  trained   - MoE, router = small classifier on the answers' own encoder states (one encoder pass)
  oracle    - the expert that owns the domain (perfect routing), for reference

    python scripts/eval_router.py                     # -> results/router_eval.json
"""
import argparse, glob, json, os, sys, time

import numpy as np
import torch
from scipy.stats import binomtest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from laya_moe.moe import MoEAgent  # noqa: E402


def score_answer(spec, ans, want):
    if spec["type"] == "noul":
        return (ans["noul"] >= 0.5) == want
    got = max(ans["probabilities"], key=ans["probabilities"].get)
    return (int(got) if spec["type"] == "score" else got) == want


def mcnemar(a, b):
    fixed, broke = int((~a & b).sum()), int((a & ~b).sum())
    return fixed, broke, (binomtest(fixed, fixed + broke, 0.5).pvalue if fixed + broke else 1.0)


def main():
    root = os.path.join(os.path.dirname(__file__), "..")
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", nargs="+", default=[os.path.join(root, "checkpoints", e) for e in ("safety", "customer_ops")])
    ap.add_argument("--router", default=os.path.join(root, "checkpoints", "router"))
    ap.add_argument("--out", default=os.path.join(root, "results", "router_eval.json"))
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(0)

    moe = MoEAgent(experts=args.experts, threshold=0.3, router=args.router)
    owner = {d: n for n, (_, m) in moe.experts.items() for d in m["domains"]}
    docs = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(root, "data", "eval", "*.json")))
            if not p.endswith("index.json")]

    # warm-up so the first timed call doesn't pay one-off costs
    moe.answer_with("general", "warm up", docs[0]["questions"])

    rows, cases, t = [], [], {"original": 0.0, "prompted": 0.0, "trained": 0.0}
    for d in docs:
        dom, qs = d["domain"], d["questions"]
        for c in d["cases"]:
            s = c["state"]
            t0 = time.perf_counter(); a_orig = moe.answer_with("general", s, qs); t["original"] += time.perf_counter() - t0
            t0 = time.perf_counter(); p_head, p_r = moe.route(s); a_pr = moe.answer_with(p_head, s, qs); t["prompted"] += time.perf_counter() - t0
            t0 = time.perf_counter(); out = moe.system_one(s, qs); t["trained"] += time.perf_counter() - t0
            a_tr, t_head = out["answers"], out["expert"]
            a_or = moe.answer_with(owner[dom], s, qs) if dom in owner else a_orig
            cases.append({"domain": dom, "id": c["id"], "expected_head": owner.get(dom, "general"),
                          "prompted_head": p_head, "prompted_top": p_r["top"],
                          "trained_head": t_head, "trained_top": out["routing"]["top"], "trained_p": out["routing"]["confidence"]})
            for q, spec in qs.items():
                w = c["expected"][q]
                rows.append({"domain": dom, "id": c["id"], "q": q, "type": spec["type"], "covered": dom in owner,
                             "original": score_answer(spec, a_orig[q], w), "prompted": score_answer(spec, a_pr[q], w),
                             "trained": score_answer(spec, a_tr[q], w), "oracle": score_answer(spec, a_or[q], w)})
        print(f"  {dom:22} done", flush=True)

    n = len(cases)
    M = {m: np.array([r[m] for r in rows]) for m in ("original", "prompted", "trained", "oracle")}
    cov = np.array([r["covered"] for r in rows])
    summary = {m: {"all": float(v.mean()), "covered": float(v[cov].mean()), "uncovered": float(v[~cov].mean())} for m, v in M.items()}
    lat = {m: 1000 * v / n for m, v in t.items()}
    route_acc = {k: float(np.mean([c[f"{k}_head"] == c["expected_head"] for c in cases])) for k in ("prompted", "trained")}
    by_dom = {}
    for dom in sorted({r["domain"] for r in rows}):
        sel = np.array([r["domain"] == dom for r in rows])
        by_dom[dom] = {m: float(v[sel].mean()) for m, v in M.items()}
        by_dom[dom]["trained_routes"] = {h: sum(1 for c in cases if c["domain"] == dom and c["trained_head"] == h)
                                         for h in ("safety", "customer_ops", "general")}
    tests = {"trained_vs_original": mcnemar(M["original"], M["trained"]),
             "trained_vs_prompted": mcnemar(M["prompted"], M["trained"])}

    print(f"\n{len(rows)} answers, {n} cases")
    print(f"{'':10} {'all':>7} {'covered':>8} {'uncov.':>7} {'ms/case':>8} {'routing':>8}")
    for m in ("original", "prompted", "trained", "oracle"):
        s = summary[m]
        print(f"{m:10} {s['all']:7.1%} {s['covered']:8.1%} {s['uncovered']:7.1%} "
              f"{lat.get(m, float('nan')):8.0f} {route_acc.get(m, float('nan')):8.1%}")
    for k, (f, b, p) in tests.items():
        print(f"{k}: fixed {f}, broke {b}, McNemar p = {p:.4f}")
    print("\nper domain (original / prompted / trained / oracle, trained routes):")
    for dom, v in by_dom.items():
        print(f"  {dom:22} {v['original']:6.1%} {v['prompted']:6.1%} {v['trained']:6.1%} {v['oracle']:6.1%}  {v['trained_routes']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"summary": summary, "latency_ms_per_case": lat, "routing_accuracy": route_acc, "by_domain": by_dom,
               "tests": {k: {"fixed": f, "broke": b, "p": p} for k, (f, b, p) in tests.items()},
               "threads": torch.get_num_threads(), "cases": cases, "answers": rows},
              open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
