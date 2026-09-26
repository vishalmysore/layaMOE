"""Score data/eval with the quantized ONNX build (what the browser runs) and compare with PyTorch.

Same routing and post-processing as the page (web/laya-core.js), run with onnxruntime on CPU, so
it measures what int8 quantization costs on the hand-labeled cases.

    python scripts/eval_onnx.py
"""
import glob, json, os, sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from laya.common import QTYPES, temp_bucket  # noqa: E402
from laya_moe.core import load_base, make_item, to_internal  # noqa: E402
from eval_moe import correct  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    onnx_dir = os.path.join(ROOT, "build", "onnx")
    man = json.load(open(os.path.join(ROOT, "build", "web", "manifest.json")))
    agent = load_base()
    tok, cfg = agent.tok, agent.cfg
    so = ort.SessionOptions()
    enc = ort.InferenceSession(os.path.join(onnx_dir, "encoder.onnx"), so, providers=["CPUExecutionProvider"])
    heads = {n: ort.InferenceSession(os.path.join(onnx_dir, f"head_{n}.onnx"), so, providers=["CPUExecutionProvider"]) for n in man["heads"]}
    router = man["router"]
    rq = {"type": "choice", "instructions": router["question"], "criteria": router["options"]}

    def run(state, questions):
        qs = [rq] + list(questions.values())
        items = [make_item(tok, cfg, state, q) for q in qs]
        n, L, K = len(items), max(len(i["ids"]) for i in items), max(len(i["markers"]) for i in items)
        ids = np.full((n, L), tok.pad_token_id, np.int64); att = np.zeros((n, L), np.int64)
        mpos = np.zeros((n, K), np.int64); mmask = np.zeros((n, K), bool)
        for r, it in enumerate(items):
            ids[r, :len(it["ids"])] = it["ids"]; att[r, :len(it["ids"])] = 1
            mpos[r, :len(it["markers"])] = it["markers"]; mmask[r, :len(it["markers"])] = True
        qt = np.array([it["qtype"] for it in items], np.int64)
        h = enc.run(None, {"input_ids": ids, "attention_mask": att})[0]

        def answers(name, rows):
            lg = heads[name].run(None, {"hidden": h[rows], "attention_mask": att[rows], "marker_pos": mpos[rows],
                                        "marker_mask": mmask[rows], "qtype": qt[rows]})[0]
            meta, out = man["heads"][name], []
            for j, r in enumerate(range(rows.start, rows.stop)):
                k = len(items[r]["markers"])
                T = meta["temperature_by_options"].get(temp_bucket(int(qt[r]), k), meta["temperature"][int(qt[r])])
                T = min(5.0, max(0.5, float(T)))  # the page clamps like layaForWeb
                z = lg[j, :k] / T
                p = np.exp(z - z.max()); p /= p.sum()
                q = to_internal(qs[r])
                if q["t"] == "choice":
                    out.append({"probabilities": dict(zip(q["crit"].keys(), p.tolist()))})
                elif q["t"] == "score":
                    out.append({"probabilities": {str(i): float(v) for i, v in enumerate(p)}})
                else:
                    out.append({"noul": float(p[1])})
            return out

        rp = answers("general", slice(0, 1))[0]["probabilities"]
        top = max(rp, key=rp.get)
        exp = router["experts"].get(top)
        chosen = exp if exp in heads and rp[top] >= router["threshold"] else "general"
        keys = list(questions)
        return chosen, dict(zip(keys, answers("general", slice(1, n)))), dict(zip(keys, answers(chosen, slice(1, n))))

    ref = {(r["domain"], r["id"], r["question"]): r for r in json.load(open(os.path.join(ROOT, "results", "moe_eval.json"), encoding="utf-8"))}
    rows, same_route = [], 0
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "eval", "*.json"))):
        if path.endswith("index.json"):
            continue
        doc = json.load(open(path, encoding="utf-8"))
        for case in doc["cases"]:
            chosen, gen, moe = run(case["state"], doc["questions"])
            for q, spec in doc["questions"].items():
                want = case["expected"][q]
                r = ref[(doc["domain"], case["id"], q)]
                same_route += chosen == r["routed_to"]
                rows.append((doc["domain"], correct(spec, gen[q], want), correct(spec, moe[q], want),
                             r["correct"]["general"], r["correct"]["moe"]))
    n = len(rows)
    pc = lambda i: f"{100 * sum(r[i] for r in rows) / n:.1f}%"
    print(f"{n} answers   same routing as PyTorch on {100 * same_route / n:.1f}% of answers")
    print(f"general head   PyTorch {pc(3)}   ONNX int8 {pc(1)}")
    print(f"MoE            PyTorch {pc(4)}   ONNX int8 {pc(2)}")


if __name__ == "__main__":
    main()
