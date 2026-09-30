"""Training data for a duplicate-question expert ("pairs") from the GLUE QQP *train* split.

A balanced sample (default 750 duplicate + 750 not) of question pairs becomes one noul question each,
with three paraphrased wordings so the head learns the task rather than one prompt. Pairs whose
questions also appear in the eval slice (QQP validation, see eval_qqp.py) are excluded.

    python scripts/gen_qqp_data.py                    # data/train/pairs/qqp.jsonl
    python scripts/cache_features.py --expert pairs
    python scripts/train_expert.py --expert pairs --lr 6e-4 --epochs 6
    python scripts/eval_qqp.py --pairs-expert checkpoints/pairs
"""
import argparse, json, os, random

import pandas as pd
from huggingface_hub import hf_hub_download

WORDINGS = ["Do question1 and question2 ask the same thing?",
            "The two questions are duplicates: answering one answers the other",
            "question1 and question2 have the same intent"]


def main():
    root = os.path.join(os.path.dirname(__file__), "..")
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=750)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(root, "data", "train", "pairs", "qqp.jsonl"))
    args = ap.parse_args()
    rnd = random.Random(args.seed)

    train = pd.read_parquet(hf_hub_download("nyu-mll/glue", "qqp/train-00000-of-00001.parquet", repo_type="dataset"))
    val = pd.read_parquet(hf_hub_download("nyu-mll/glue", "qqp/validation-00000-of-00001.parquet", repo_type="dataset"))
    eval_slice = val.groupby("label", group_keys=False).sample(n=750, random_state=42)
    eval_q = set(eval_slice.question1.str.lower().str.strip()) | set(eval_slice.question2.str.lower().str.strip())
    clean = train[~train.question1.str.lower().str.strip().isin(eval_q) & ~train.question2.str.lower().str.strip().isin(eval_q)]
    sample = clean.groupby("label", group_keys=False).sample(n=args.per_class, random_state=args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in sample.itertuples():
            q = {"type": "noul", "instructions": rnd.choice(WORDINGS)}
            f.write(json.dumps({"domain": "qqp-duplicates", "expert": "pairs",
                                "state": {"question1": r.question1, "question2": r.question2},
                                "questions": {"duplicate": q}, "expected": {"duplicate": bool(r.label)}},
                               ensure_ascii=False) + "\n")
    print(f"wrote {len(sample)} pairs ({int(sample.label.sum())} duplicate) to {args.out}; "
          f"{len(train) - len(clean)} train pairs dropped for sharing a question with the eval slice")


if __name__ == "__main__":
    main()
