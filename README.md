# layaMOE

**Domain experts on top of Laya.** One shared, frozen [laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) encoder, several small expert heads fine-tuned for a group of domains, a router that picks one per request, and the original head as a fallback.

**Result:** on 108 hand-labeled cases (312 answers) the MoE scores **67.3%** against **59.6%** for the plain laya-typed-decisions model, with no loss on the domains that have no expert.

**Live demo:** https://vishalmysore.github.io/layaMOE/ (runs in the browser with ONNX Runtime Web)

**Compare it yourself:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vishalmysore/layaMOE/blob/main/notebooks/laya_vs_moe_comparison.ipynb) runs the original Laya and the MoE side by side on the eval set, with per-domain results, paired significance tests, routing analysis and calibration ([notebooks/laya_vs_moe_comparison.ipynb](notebooks/laya_vs_moe_comparison.ipynb)).

## Why not a "real" MoE?

Mixture-of-Experts LLMs (DeepSeek-V3, the nanoMoE tutorial, ...) replace every feed-forward block with N experts and a per-token top-k router, trained from scratch with load-balancing and router z-losses. For Laya that is the wrong fit:

- Laya is an encoder plus a small decision head, not a generator. Per-token experts learn their own specialisations; they would not line up with "safety" or "customer operations".
- N copies of every feed-forward block multiply the download size, and a browser runtime (ONNX Runtime Web) has no sparse MoE kernel, so every expert would be computed anyway.

So this project uses **sequence-level routing over expert heads**:

```
           input text + typed question
                      │
          ┌───────────▼────────────┐
          │  ModernBERT encoder    │  395M params, frozen, shared, downloaded once
          │  (from laya-typed-...) │
          └───────────┬────────────┘
                      │ hidden states
      router ─────────┼─────────────────────┐
  (one choice question│                     │
   on the base model) ▼                     ▼
   ┌──────────┐  ┌──────────────┐  ┌──────────────┐
   │ general  │  │ safety head  │  │ customer_ops │   each head: 26.5M params
   │ (original│  │ guardrails + │  │ tickets +    │   (type embedding, 2 transformer
   │  head)   │  │ moderation   │  │ delivery +   │    layers, scorer), ~27 MB at int8
   └──────────┘  └──────────────┘  │ email        │
                                   └──────────────┘
```

The router asks the base model one choice question, "what kind of text is this?", over fine-grained kinds (agent action, public comment, support ticket, delivery issue, work email, IT alert, patient message, product review, sales inquiry, other). Each kind maps to an expert or to the general head. Fine-grained kinds matter: a first router that asked "which expert?" with three broad options sent 9 of 12 guardrail cases to the general head and half of the patient messages to customer_ops. The inputs and outputs are identical to Laya's `system_one`, and each expert has its own fitted calibration temperatures.

## Experts

| Expert | Domains | Why |
|---|---|---|
| `safety` | agent-guardrails, content-moderation | weakest domain for the base model and the costliest mistakes |
| `customer_ops` | support-tickets, delivery-exceptions, email-triage | the same route / urgency / mood pattern, 47-67% today |
| `general` | everything else | the original laya-typed-decisions head |

Patient messages are deliberately left on the general head: a clinical routing expert needs clinically reviewed labels.

## Results

Hand-labeled cases in `data/eval` (108 cases, 312 answers, 9 domains; PyTorch fp32 on CPU). Three answers per question:
**general** = the original laya-typed-decisions head, **oracle** = the expert that owns the domain (perfect routing),
**moe** = what the router actually picked.

| Domain | general | oracle | moe | routed to |
|---|---|---|---|---|
| agent-guardrails | 39.6% | 66.7% | **66.7%** | safety 12/12 |
| content-moderation | 75.0% | 83.3% | 75.0% | safety 3, general 8, customer_ops 1 |
| delivery-exceptions | 47.2% | 63.9% | **63.9%** | customer_ops 11/12 |
| email-triage | 52.8% | 58.3% | 50.0% | customer_ops 5, general 5, safety 2 |
| support-tickets | 66.7% | 80.6% | **83.3%** | customer_ops 11/12 |
| it-incidents (no expert) | 61.1% | | 61.1% | general 12/12 |
| patient-messages (no expert) | 83.3% | | 83.3% | general 12/12 |
| product-reviews (no expert) | 77.8% | | 77.8% | general 9, customer_ops 3 |
| sales-leads (no expert) | 41.7% | | 41.7% | general 12/12 |
| **domains with an expert** | 55.2% | 70.3% | **67.7%** | |
| **domains without one** | 66.7% | | **66.7%** | |
| **All** | 59.6% | 68.9% | **67.3%** | |

By question type (moe vs general): score 59.4% vs 41.7%, yes/no 78.3% vs 68.3%, choice 61.5% vs 66.7%.

Read these with care:

- 12 cases per domain: one case is 3-4 answers, so per-domain differences of under ~10 points are noise.
- The router's kind list was written after the first router failed on this eval set (one revision, generic descriptions, no per-case tuning). That is still a look at the test set; judge the router on your own data.
- Where the experts help most is score questions (the base model answers "Medium" to almost everything) and yes/no questions. Choice questions got slightly worse, mostly from email and moderation cases routed to the wrong expert or where the expert's rules differ from the eval labels.
- Individual answers can still be clearly wrong. Example (also a demo preset): for `DROP TABLE invoices` on production with no backup, the safety expert says destructive 65% but needs_human only 23%. Use the confidence values and keep a person in the loop.
- Remaining weak spots: moderation and email routing (the router often reads them as "other"), and delivery `action` (43% even on the expert's own validation data).

Expert training (validation split of the synthetic data, not the eval set):

| Expert | Items | Base head | Expert | Settings |
|---|---|---|---|---|
| safety | 3,120 train / 380 val | 58.4% | 73.4% | lr 6e-4, 6 epochs, ~45 min on CPU |
| customer_ops | 4,011 train / 489 val | 57.3% | 75.9% | lr 6e-4, 6 epochs, ~100 min on CPU |

Things learned along the way:

- The split is exact: running the frozen encoder and then a copied head reproduces the original model's logits (max difference 0.0); fp16-cached features differ by 3e-4.
- The head needs a much higher learning rate than a full fine-tune: at 3e-5 it barely moves; at 6e-4 it learns steadily and was still improving at epoch 6.
- Training only the head is CPU-bound (6-17 minutes per epoch on a 13th-gen i9). A free GPU notebook would make this minutes and would make LoRA on the encoder practical.

## Training data

`scripts/gen_train_data.py` builds synthetic cases from slots (action, environment, safety net, tone, deadline, ...) and computes each label from the slots by explicit rules, so labels are consistent by construction. Question wordings are paraphrased (3 per question, plus alternative option/level texts) and choice options are shuffled, so an expert learns the domain rather than one prompt. 500 states per domain, about 8,500 question items in total.

The hand-written evaluation cases in `data/eval` are never used for training: any generated state that shares a word 5-gram with an eval case is dropped. The generated data is synthetic and template-based, so the eval set, written separately, is the real test. With 12 cases per domain, differences of one or two answers are noise.

## Run it

Python 3.12, CPU is enough.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python scripts/eval_baseline.py                      # base model on data/eval
python scripts/gen_train_data.py                     # data/train/<expert>/<domain>.jsonl
python scripts/cache_features.py --expert safety     # frozen encoder once -> cache/safety.pt (~400 MB)
python scripts/train_expert.py --expert safety --lr 6e-4 --epochs 6
python scripts/cache_features.py --expert customer_ops
python scripts/train_expert.py --expert customer_ops --lr 6e-4 --epochs 6
python scripts/eval_moe.py                           # general vs oracle routing vs real routing
```

Use it from Python:

```python
from laya_moe.moe import MoEAgent

moe = MoEAgent(experts=["checkpoints/safety", "checkpoints/customer_ops"])
out = moe.system_one("Agent plan: run `DROP TABLE invoices` on the production database.", {
    "destructive": {"type": "noul", "instructions": "The action is destructive and cannot be undone"},
    "risk": {"type": "score", "instructions": "How risky is this action?", "criteria": ["Low", "Medium", "High", "Critical"]},
})
print(out["expert"], out["routing"], out["answers"])
```

## Layout

| Path | What it is |
|---|---|
| `laya_moe/core.py` | load the base checkpoint, `ExpertHead` (everything after the encoder), input building, save/load experts |
| `laya_moe/moe.py` | `MoEAgent`: router, expert heads, general fallback, calibrated answers |
| `scripts/gen_train_data.py` | synthetic rule-labeled training data |
| `scripts/cache_features.py` | run the frozen encoder once and cache hidden states |
| `scripts/train_expert.py` | train one expert head from the cache, keep the best epoch, fit temperatures |
| `scripts/eval_baseline.py` | score a plain Laya checkpoint on `data/eval` |
| `scripts/eval_moe.py` | compare general / oracle-routed / routed answers on `data/eval` |
| `scripts/export_web.py` | ONNX export (encoder + heads), int8 quantization, check against PyTorch, packaging |
| `scripts/upload_hf.py` | upload experts or the browser build to Hugging Face |
| `notebooks/laya_vs_moe_comparison.ipynb` | Colab notebook: original Laya vs MoE on `data/eval` (stats, routing, calibration) |
| `web/`, `scripts/prepare_site.mjs` | the browser demo and its site builder |
| `results/moe_eval.json` | every eval answer (general, oracle, moe) with routing |
| `data/eval/` | 108 hand-labeled cases in 9 domains (from layaForWeb) |
| `data/train/` | generated training data |

Articles: [docs/article.md](docs/article.md) · [Laya Vs Jev - Laya Deep Dive](docs/laya-vs-jev-laya-deep-dive.md) (Laya vs Jev internals, plus a full re-run of the comparison notebook with screenshots).

Not in git: `.venv/`, `cache/` (encoder features), `checkpoints/` (expert weights, on Hugging Face at [VishalMysore/layaMOE](https://huggingface.co/VishalMysore/layaMOE)), `build/` (browser files, at [VishalMysore/laya-moe-web](https://huggingface.co/VishalMysore/laya-moe-web)).

## Browser build

`scripts/export_web.py` exports the encoder and every head as separate ONNX graphs (weight-only int8), checks them against PyTorch, and splits the weights into 24 MiB parts with SHA-256 hashes. The page (`web/`) downloads the encoder once, runs it once per request for the router question and all user questions together, then runs the chosen head (and optionally the general head, to compare) on the same hidden states.

```
python scripts/export_web.py            # build/web/
python scripts/upload_hf.py --web       # -> huggingface.co/VishalMysore/laya-moe-web
npm ci && node scripts/prepare_site.mjs --local-model build/web && python serve.py   # local test
```

GitHub Actions (`.github/workflows/pages.yml`) only assembles and deploys the page; it never trains or converts models.

`scripts/eval_onnx.py` runs the hand-labeled eval through the int8 ONNX files the page uses: MoE 67.9% (PyTorch 67.3%), general head 58.7% (PyTorch 59.6%), same routing as PyTorch on every case. In the browser (WASM, 4 threads, 13th-gen i9) a request with three questions takes about 5 s: ~4.5 s encoder, ~0.6 s router and heads.

## Next

- Train the router instead of prompting it (moderation and email routing are the weak spots).
- More and more varied training data; LoRA on the top encoder layers on a GPU.
- More experts (IT incidents, sales leads).

## License

Apache-2.0 (see `LICENSE`). Built on laya-typed-decisions by ConvAI Innovations and ModernBERT-large by Answer.AI and LightOn, both Apache-2.0; see `NOTICE.md`. Unofficial, not affiliated with ConvAI Innovations.
