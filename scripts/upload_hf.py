"""Upload trained experts or the browser build to Hugging Face.

    python scripts/upload_hf.py --experts            # checkpoints/<expert>/ -> VishalMysore/layaMOE/<expert>/
    python scripts/upload_hf.py --web                # build/web/ -> VishalMysore/laya-moe-web

Uses HF_TOKEN from the environment or the saved `huggingface-cli login` token.
"""
import argparse, glob, json, os, shutil, tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent

EXPERTS_CARD = """---
license: apache-2.0
base_model: convaiinnovations/laya-typed-decisions
tags: [decision-model, mixture-of-experts, laya]
---
# Laya MoE expert heads

Domain expert heads for [layaMOE](https://github.com/vishalmysore/layaMOE). Each folder holds one decision head
(type embedding, 2 transformer layers, scorer; 26.5M parameters) fine-tuned from the head of
[convaiinnovations/laya-typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) on synthetic,
rule-labeled domain data, with its own calibration temperatures in `expert.json`. The encoder is not included:
load it from the base checkpoint, which is shared by all experts.

{table}

```python
from laya_moe.moe import MoEAgent   # github.com/vishalmysore/layaMOE
from huggingface_hub import snapshot_download
d = snapshot_download("VishalMysore/layaMOE")
moe = MoEAgent(experts=[f"{{d}}/safety", f"{{d}}/customer_ops"])
```

Modified derivative of laya-typed-decisions (Apache-2.0, Copyright ConvAI Innovations); unofficial and not
affiliated with ConvAI Innovations. The training data is synthetic; evaluate on your own cases before relying on it.
See `LICENSE` and `NOTICE.md`.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", action="store_true")
    ap.add_argument("--web", action="store_true")
    ap.add_argument("--experts-repo", default="VishalMysore/layaMOE")
    ap.add_argument("--web-repo", default="VishalMysore/laya-moe-web")
    args = ap.parse_args()
    api = HfApi(token=os.environ.get("HF_TOKEN"))

    if args.experts:
        with tempfile.TemporaryDirectory() as tmp:
            rows = ["| Expert | Domains | Validation acc. (base head -> expert) |", "|---|---|---|"]
            for d in sorted(glob.glob(str(ROOT / "checkpoints" / "*"))):
                meta = json.load(open(Path(d) / "expert.json"))
                shutil.copytree(d, Path(tmp) / meta["expert"])
                t = meta.get("train", {})
                rows.append(f"| `{meta['expert']}` | {', '.join(meta['domains'])} | {t.get('val_acc_base_head', 0):.1%} -> {t.get('val_acc', 0):.1%} |")
            for f in ("LICENSE", "NOTICE.md"):
                shutil.copy(ROOT / f, Path(tmp) / f)
            (Path(tmp) / "README.md").write_text(EXPERTS_CARD.format(table="\n".join(rows)), encoding="utf-8")
            api.upload_folder(repo_id=args.experts_repo, folder_path=tmp, commit_message="Upload expert heads")
        print("uploaded experts to", args.experts_repo)

    if args.web:
        api.upload_folder(repo_id=args.web_repo, folder_path=str(ROOT / "build" / "web"), commit_message="Upload browser build")
        print("uploaded browser build to", args.web_repo)


if __name__ == "__main__":
    main()
