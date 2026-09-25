# layaMOE

**Domain experts on top of Laya.** One shared, frozen [laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) encoder, several small expert heads fine-tuned for a group of domains, a router that picks one per request, and the original head as a fallback.

> Status: work in progress. The pipeline runs end to end; the first expert is still being tuned. Numbers below are what has been measured so far, nothing more.

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

The router asks the base model one choice question ("what kind of input is this?"). If an expert wins with probability at or above the threshold (default 0.6), that expert's head answers; otherwise the untouched general head does. The inputs and outputs are identical to Laya's `system_one`, and each expert has its own fitted calibration temperatures.

## Experts

| Expert | Domains | Why |
|---|---|---|
| `safety` | agent-guardrails, content-moderation | weakest domain for the base model and the costliest mistakes |
| `customer_ops` | support-tickets, delivery-exceptions, email-triage | the same route / urgency / mood pattern, 47-67% today |
| `general` | everything else | the original laya-typed-decisions head |

Patient messages are deliberately left on the general head: a clinical routing expert needs clinically reviewed labels.

## Results so far

**Baseline**, laya-typed-decisions on the 108 hand-labeled cases in `data/eval` (312 answers, PyTorch fp32, CPU):

| Domain | Correct |
|---|---|
| agent-guardrails | 39.6% |
| content-moderation | 75.0% |
| delivery-exceptions | 47.2% |
| email-triage | 52.8% |
| it-incidents | 61.1% |
| patient-messages | 83.3% |
| product-reviews | 77.8% |
| sales-leads | 41.7% |
| support-tickets | 66.7% |
| **All** | **59.6%** |

By type: choice 66.7%, score 41.7% exact (80.2% within one level), yes/no 68.3%.

**Safety expert, first run** (lr 2e-4, 3 epochs, validation split of the synthetic data, not the eval set): 58.4% → 65.3%. Guardrail yes/no 55% → 66%, guardrail risk score 21% → 38%. It is still underfitting; a run at a higher learning rate is in progress. Results on the hand-labeled eval set will be added once both experts are trained.

Things learned along the way:

- The split is exact: running the frozen encoder and then a copied head reproduces the original model's logits (max difference 0.0), and fp16-cached features differ by 3e-4.
- The head needs a much higher learning rate than a full fine-tune: at 3e-5 it barely moves; at 3e-4 it can overfit 64 examples to 95% in 40 steps.
- Training only the head is CPU-bound at about 6-7 minutes per epoch for 3,000 items on a 13th-gen i9. A free GPU notebook would make this minutes, and would also make LoRA on the encoder practical if head-only training plateaus.

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
python scripts/train_expert.py --expert customer_ops
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
| `data/eval/` | 108 hand-labeled cases in 9 domains (from layaForWeb) |
| `data/train/` | generated training data |

Not in git: `.venv/`, `cache/` (encoder features), `checkpoints/` (expert weights; these will go to a Hugging Face model repo).

## Next

1. Finish tuning the safety expert; train customer_ops.
2. Evaluate on `data/eval`: general head vs perfect routing vs the real router.
3. If the experts help: upload expert heads to Hugging Face.
4. Browser build: export the encoder and each head as separate ONNX graphs, so the page downloads the encoder once and each expert for ~27 MB.

## License

Apache-2.0 (see `LICENSE`). Built on laya-typed-decisions by ConvAI Innovations and ModernBERT-large by Answer.AI and LightOn, both Apache-2.0; see `NOTICE.md`. Unofficial, not affiliated with ConvAI Innovations.
