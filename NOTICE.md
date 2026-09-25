# Notice

layaMOE is an unofficial experiment that adds domain-expert heads to the laya-typed-decisions checkpoint. It is not affiliated with or endorsed by ConvAI Innovations, Answer.AI or LightOn.

## This project

The scripts, training code, generated training data and evaluation code in this repository are Copyright 2026 vishalmysore and licensed under the Apache License, Version 2.0 (see `LICENSE`). The evaluation cases in `data/eval` are copied from the author's layaForWeb repository (same license).

## Laya (model weights, tokenizer, reference code)

- Source: https://huggingface.co/convaiinnovations/laya-typed-decisions and https://github.com/NandhaKishorM/laya (PyPI: `laya`)
- Copyright: ConvAI Innovations
- License: Apache License, Version 2.0

**Changes made in this project** (Apache-2.0 section 4b):

- The decision head (type embedding, 2 transformer layers, scorer, act head) of laya-typed-decisions was copied and fine-tuned on synthetic domain data to make each expert. The encoder is unchanged.
- Calibration temperatures were refitted for each expert.

Expert weights produced by this project are modified derivatives of laya-typed-decisions and are not the original release.

## ModernBERT

Laya's encoder comes from ModernBERT-large by Answer.AI and LightOn, released under the Apache License, Version 2.0. https://huggingface.co/answerdotai/ModernBERT-large
