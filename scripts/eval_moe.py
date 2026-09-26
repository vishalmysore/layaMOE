"""Score the hand-written cases in data/eval with the MoE and compare with the base head.

For every case it reports three answers:
  general  the untouched laya-typed-decisions head (the baseline)
  oracle   the expert that owns the case's domain (upper bound: perfect routing)
  moe      whatever the router picked (the real system)

    python scripts/eval_moe.py
    python scripts/eval_moe.py --experts checkpoints/safety checkpoints/customer_ops --threshold 0.3 --out results/moe.json
"""
import argparse, glob, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from laya_moe.moe import MoEAgent  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def correct(spec, ans, want):
    if spec["type"] == "choice":
        return max(ans["probabilities"], key=ans["probabilities"].get) == want
    if spec["type"] == "noul":
        return (ans["noul"] >= 0.5) == want
    return int(max(ans["probabilities"], key=ans["probabilities"].get)) == want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", nargs="*", default=sorted(glob.glob(os.path.join(ROOT, "checkpoints", "*"))))
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--out")
    args = ap.parse_args()

    moe = MoEAgent(args.experts, threshold=args.threshold)
    owner = {d: name for name, (_, meta) in moe.experts.items() for d in meta["domains"]}
    print("experts:", {k: v[1]["domains"] for k, v in moe.experts.items()}, "\n")

    rows, dump, route_tab = [], [], {}
    t0 = time.time()
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "eval", "*.json"))):
        if path.endswith("index.json"):
            continue
        doc = json.load(open(path, encoding="utf-8"))
        dom, qs = doc["domain"], doc["questions"]
        for case in doc["cases"]:
            chosen, routing = moe.route(case["state"])
            route_tab.setdefault(dom, {}).setdefault(chosen, 0)
            route_tab[dom][chosen] += 1
            ans = {"general": moe.answer_with("general", case["state"], qs)}
            if dom in owner:
                ans["oracle"] = moe.answer_with(owner[dom], case["state"], qs)
            ans["moe"] = ans["general"] if chosen == "general" else (
                ans["oracle"] if owner.get(dom) == chosen else moe.answer_with(chosen, case["state"], qs))
            ans.setdefault("oracle", ans["general"])
            for q, spec in qs.items():
                want = case["expected"][q]
                ok = {m: correct(spec, ans[m][q], want) for m in ("general", "oracle", "moe")}
                rows.append((dom, spec["type"], ok))
                dump.append({"domain": dom, "id": case["id"], "question": q, "expected": want, "routed_to": chosen,
                             "routing": routing, "correct": ok, "answers": {m: ans[m][q] for m in ans}})

    def pct(sel, m):
        return f"{100 * sum(r[2][m] for r in sel) / len(sel):5.1f}%" if sel else "    -"

    print(f"{'domain':22} {'n':>4} {'general':>8} {'oracle':>8} {'moe':>8}   routed to")
    for dom in sorted({r[0] for r in rows}):
        sel = [r for r in rows if r[0] == dom]
        print(f"{dom:22} {len(sel):4d} {pct(sel, 'general'):>8} {pct(sel, 'oracle'):>8} {pct(sel, 'moe'):>8}   {route_tab[dom]}"
              + ("   [expert: %s]" % owner[dom] if dom in owner else ""))
    cov = [r for r in rows if r[0] in owner]
    unc = [r for r in rows if r[0] not in owner]
    print(f"{'covered domains':22} {len(cov):4d} {pct(cov, 'general'):>8} {pct(cov, 'oracle'):>8} {pct(cov, 'moe'):>8}")
    print(f"{'other domains':22} {len(unc):4d} {pct(unc, 'general'):>8} {pct(unc, 'oracle'):>8} {pct(unc, 'moe'):>8}")
    print(f"{'ALL':22} {len(rows):4d} {pct(rows, 'general'):>8} {pct(rows, 'oracle'):>8} {pct(rows, 'moe'):>8}\n")
    for t in ("choice", "score", "noul"):
        sel = [r for r in rows if r[1] == t]
        print(f"  {t:7} {len(sel):4d}  general {pct(sel, 'general')}  oracle {pct(sel, 'oracle')}  moe {pct(sel, 'moe')}")
    print(f"\n{time.time() - t0:.0f}s")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(dump, open(args.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
