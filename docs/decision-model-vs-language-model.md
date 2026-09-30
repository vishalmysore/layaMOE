---
title: "Decision Model vs Language Model: OpenAI's Decisions API, Laya and the New Breed of System One Models"
description: "What decision models are, how they differ from language models, what OpenAI's new Decisions API adds, and how open models like Laya compare, with architectures, benchmarks and a practical guide to choosing."
keywords: [decision model, decision model vs language model, OpenAI Decisions API, GPT-6 Luna, System One model, Laya, Jev, Kev, Von, typed decisions, calibrated probabilities, AI agent routing, classification, LLM alternatives]
---

# Decision Model vs Language Model: OpenAI's Decisions API, Laya and the New Breed of System One Models

*Language models write. Decision models choose. With OpenAI entering the category, here is how the two differ inside, where each belongs in a system, and how the open model Laya compares.*

---

## TL;DR

- A **decision model** takes some context and a question whose answers you define up front, and returns a probability over those answers in one pass. A **language model** generates text token by token, and your code has to parse the result.
- In a few weeks, decision models have grown from one product into a **whole category**: hosted APIs (TypeSafe's **Jev**, OpenAI's new **Decisions API**), open encoders (**Laya**, **Von**), open LLM fine-tunes (**Kev**, **Decider**, **Tev1**), and training-free wrappers over existing LLMs.
- OpenAI's **Decisions API**, announced at DevDay 2026 in limited preview, runs a version of **GPT-6 Luna** on "user-defined questions with finite pre-defined answers", takes **text or images**, and is reported at about **150 ms per decision**.
- **Laya** is the open counterpoint: a 421M-parameter ModernBERT encoder with a typed decision head. It answers choice, score and yes/no questions in tens of milliseconds on a GPU, runs in a browser tab, and can be specialised with small expert heads.
- The rule of thumb: **if you can write down the answer space before the call, use a decision model. If you need the model to write, explain or plan, use a language model.** In most real systems you use both.

---

## The new breed of decision models

For years the default way to make an AI system *decide* something was to ask a chat model and parse what it wrote. September 2026 changed that. TypeSafe AI shipped Jev as a dedicated "System One" decision model. Open alternatives followed within days, and on September 29 OpenAI announced its own **Decisions API**. Community lists now track more than a dozen System One models ([awesome-system-one](https://github.com/yanng981/awesome-system-one), [awesome-jev-alternatives](https://github.com/mturac/awesome-jev-alternatives)).

The "System One" name comes from Kahneman's fast, intuitive thinking, as opposed to slow, deliberate System Two. A System One model is not meant to reason through a problem. It makes one quick, bounded judgment and reports how sure it is.

The models fall into four families that share a contract but differ in how they are built:

| Family | Examples | How it decides | Runs where |
|---|---|---|---|
| **Hosted, frontier-derived** | OpenAI Decisions API (GPT-6 Luna), TypeSafe Jev | Undisclosed internals behind a typed API | Provider's cloud |
| **Encoder + decision head** | Laya (ModernBERT-large / mmBERT), Von (ModernBERT), GLiNER2.5-Decide | Bidirectional encoder reads text and options together, then a head scores each option | CPU, GPU, even the browser |
| **LLM backbone + readout head** | Kev (Qwen + LoRA), Decider (Qwen, up to MoE), Tev1-4B (Qwen3.5), NanoJev | A causal LM prefills once, then a small head reads option hidden states. No decoding | GPU or laptop |
| **Training-free wrappers** | SemIf, AnyJev | Read and renormalise the logits a normal LLM gives each option's tokens | Wherever the LLM runs |

What unites them is the **contract**: context in, typed questions in, probabilities out. The three question types most of them share are:

- **choice**: pick one label from a list supplied at request time,
- **score**: place the input on an ordered scale,
- **yes/no** (called *noul* by Jev and Laya): the probability that a statement is true.

## Decision model vs language model: what actually differs

### 1. The output space is fixed before inference

A language model's output space is its entire vocabulary, sampled again at every step. Ask it to classify a ticket as billing, technical or account, and it may say "Billing (likely)", "billing/technical" or a paragraph of reasoning. Structured output modes help with *format*, but the model is still generating.

A decision model fixes the answer set **before** the forward pass. It is structurally impossible for it to return a fourth option to a three-option question. OpenAI puts it as "finite pre-defined answers", and Laya has no generation path at all.

```
language model:   prompt → tokens → text → parse → validate → retry → action
decision model:   context + typed question → probabilities over your options → policy → action
```

### 2. One pass instead of a loop

Generation is sequential: one forward pass per token until a stop token. A decision model reads the input once and scores every option from that one pass. Different families get there in different ways:

- **Restricted softmax** (as publicly described for Jev): take the logits for only the declared options and normalise over those few entries.
- **Per-option markers** (Laya): each option gets its own `[MASK]` position in the input, a scorer reads one logit per marker, and a softmax runs over that question's options.
- **Pointer readout** (Kev): after the options, a `<decide>` token's hidden state is compared with each option's hidden state, then softmaxed.

All three give you a full distribution, not just a winner. Several questions about the same context can be answered from one pass.

### 3. Probability is the product, not a side effect

A language model can *say* "I'm 90% sure", but that number is itself generated text. Decision models are trained so that the probability means something. Jev, Laya and Von describe training toward calibration: Laya uses reinforcement learning with **strictly proper scoring rules** (log, spherical, and a ranked probability score for ordinal questions), which pays the most when the model reports its true belief. A calibrated model's 0.8 answers are right about 80% of the time *in aggregate*, so your code can act automatically above a threshold and send the rest to review.

### 4. What they can't do

Decision models don't write replies, summarise, explain or plan. They handle multi-step hidden reasoning, arithmetic and date math poorly, and irrelevant context lowers their accuracy. A valid answer can still be wrong: a schema rules out answers outside the list, not wrong answers inside it.

### Side by side

| | Language model | Decision model |
|---|---|---|
| Output | Free text (optionally shaped as JSON) | A probability for each allowed answer |
| Answer space | Whole vocabulary | Declared per request |
| Inference | One pass per generated token | One pass per request |
| Many questions, one input | Longer prompt, longer output | Answered together from the same read |
| Confidence | Generated or inferred after the fact | Trained toward calibration |
| Best at | Drafting, explaining, planning, tool orchestration | Routing, classification, triage, gating, agent next-step choice |
| Failure mode | Format drift, off-list answers, hallucinated content | Confidently choosing the wrong listed option |

## OpenAI's Decisions API: what we know

From OpenAI's [DevDay 2026 recap](https://openai.com/index/devday-2026-recap/) and launch coverage:

- **Purpose.** "Decisions API enables real-time decision-making by focusing Luna's intelligence on a specific set of user-defined questions with finite pre-defined answers."
- **Engine.** A version of **GPT-6 Luna**, OpenAI's efficient GPT-6 tier. This makes it the first decision model derived from a frontier LLM family.
- **Inputs.** Context as **text or images**. Image context is the clearest difference from the text-only open models.
- **Use cases.** Content classification, request routing, and choosing an agent's next action.
- **Speed.** About **150 ms per decision**, reported as roughly **10× faster** than calling GPT-6 Luna through the regular API. These are launch claims, not a service-level guarantee.
- **Status.** Limited preview, with broader availability "expected in the coming days".

What is **not** public yet: the exact request and response schema, whether several questions can share one call, whether score and yes/no questions exist as first-class types or only choice, what the returned score means and whether it is calibrated, and any published accuracy figures. Treat third-party payload examples as illustrative and keep the provider behind a thin adapter until the official reference settles.

## Laya compared: open weights vs OpenAI's Decisions API vs Jev

[Laya](https://huggingface.co/convaiinnovations/laya-typed-decisions) is the most-studied open decision model, and it makes a useful reference point because every layer can be inspected:

```
[CLS] <type> question [SEP] [MASK] option1 [MASK] option2 … [SEP] context [SEP]
            │
   ModernBERT-large encoder (395M)
            │
   + question-type embedding → 2 transformer layers (decision head, ~26.5M)
            │
   one logit per option [MASK] → temperature → softmax
```

| | **OpenAI Decisions API** | **TypeSafe Jev** | **Laya** |
|---|---|---|---|
| Weights | Closed | Closed | Open, Apache-2.0 |
| Engine | Version of GPT-6 Luna | Undisclosed | ModernBERT-large + decision head (421M); mmBERT multilingual variant (322M) |
| Input | Text or images | Text, JSON objects, text arrays | Text |
| Question types | Finite predefined answers (full schema in preview) | choice, score, noul | choice, score, noul |
| Many questions per call | To be confirmed | Yes | Yes |
| Context | Not yet published | ~32K input tokens | 512–1,024 tokens (multilingual up to 8,192) |
| Many-label choice | Not yet published | Strong (up to 255 options) | Weak past ~20 options |
| Calibration | Not yet published | Trained with RLCD | RLCD with proper scoring rules, plus per-type temperature refit |
| Reported speed | ~150 ms per decision (launch claim) | Tens to hundreds of ms per call (varies by report) | 32.8–39.5 ms per question on a Tesla T4 |
| Runs locally / offline | No | No | Yes, including in the browser via ONNX Runtime Web |
| Can be specialised | Not yet published | Not self-serve | Yes: fine-tune, or add expert heads |

A few observations:

- **Images are OpenAI's edge.** No open decision model in the current wave takes image context natively. If your decision depends on a screenshot or a photo, the Decisions API is the only one of the three that does it out of the box.
- **Long context and many labels favour the hosted models.** On Banking77 (77 intents), Laya's model card reports 0.425 accuracy against 0.870 for Jev. All of Laya's options share a fixed token budget in its head, so with dozens of labels each one gets only a few tokens. Split large label sets into a coarse-to-fine hierarchy.
- **Laya wins on locality and control.** It runs on your hardware or in a browser tab, so no data leaves the machine, and you can see, measure and change every weight.
- **Laya needs specialising.** Laya's own card reports the base checkpoint close to chance zero-shot (0.362 vs a 0.318 random baseline) on the typed-decisions benchmark. The fine-tuned checkpoint reaches 0.766 there, against 0.727 published for Jev.

### What specialising Laya looks like in practice

Because the weights are open, you can go further than fine-tuning. In [layaMOE](https://github.com/vishalmysore/layaMOE) I kept Laya's encoder frozen, added two small expert heads (26.5M parameters each, trained on a laptop CPU), and used one extra Laya question, "what kind of text is this?", as a router. On 108 hand-labeled cases (312 answers, 9 domains), the result was:

| | Accuracy |
|---|---|
| Original `laya-typed-decisions` | 59.6% |
| Laya + expert heads (MoE) | **67.3%** |
| Same, with perfect routing | 68.9% |

The gain is +7.7 points (95% CI +3.6 to +11.8, McNemar p = 0.0009). Domains with no expert were unchanged. The full write-up and notebook are in [Laya Vs Jev - Laya Deep Dive](laya-vs-jev-laya-deep-dive.md). The point for this article: with an open decision model, "make it better at *my* decisions" is an engineering task you can do yourself.

## Where each belongs in a real system

Decision models don't replace language models. Most production agents end up with both, in separate layers:

```
 ┌─────────────────────────────────────────────────────────┐
 │ Orchestrator   task state, retries, loop control        │
 ├─────────────────────────────────────────────────────────┤
 │ Language model understands goals, plans, writes text    │  ← System Two
 ├─────────────────────────────────────────────────────────┤
 │ Decision model route? risk? priority? done? next step?  │  ← System One
 ├─────────────────────────────────────────────────────────┤
 │ Policy         permissions, allowlists, confirmations   │  ← deterministic
 ├─────────────────────────────────────────────────────────┤
 │ Actions + audit                                         │
 └─────────────────────────────────────────────────────────┘
```

The decision layer handles the high-volume, narrow judgments the language model would otherwise be asked for thousands of times a day. It **never replaces the policy layer**. A decision model may say an action looks safe, but only code or a person grants permission to run it:

> **executable action = model judgment ∩ deterministic policy**

### A simple chooser

Use a **decision model** when:
- the possible outcomes are known before the call (queues, labels, levels, allow/confirm/block),
- the decision repeats at high volume and latency matters,
- you want a probability to gate automation on,
- you need several judgments about the same input at once.

Use a **language model** when:
- the output is language for a person (a reply, a summary, an explanation),
- the task needs planning, tool orchestration or multi-step reasoning,
- the answer space can't be enumerated in advance.

Then pick *which* decision model:
- **OpenAI Decisions API**: image context, or you want decisions from the same frontier family and platform as your other OpenAI calls.
- **Jev**: long documents, many labels, and a documented typed contract available today.
- **Laya, Von, Kev and other open models**: data must stay on your hardware or in the browser, you need offline or edge inference, or you want to fine-tune and inspect the model.

## Evaluating any decision model before you trust it

Whichever model you choose, the discipline is the same:

1. **Start with one reversible decision** and a set of real, labeled historical examples.
2. **Include a fallback answer** (`none_of_the_above`, `manual_review`) so unusual inputs aren't forced into a misleading label.
3. **Measure calibration on your own data**, not just accuracy. In the layaMOE run, the experts raised accuracy but also raised calibration error (ECE 0.055 → 0.088). At a 0.9 threshold the original model auto-acted on 12 of 312 answers (all correct), and the MoE on 63 (93.7% correct). Specialisation made the model more decisive, and not every extra decision was right.
4. **Run in shadow mode** next to human decisions before letting anything act, especially for money movement, access control, deletion or safety.
5. **Version everything**: question wording, option set, threshold, policy and model ID. Changing an option changes what past results mean.
6. **Treat timeouts and errors as "unknown"**, never as a confident no.

## FAQ

**What is a decision model?**
A model that takes context plus a question with predefined answers and returns a probability for each answer in one pass, without generating text. Common question types are choice, score and yes/no.

**How is a decision model different from a large language model?**
An LLM generates text token by token over its whole vocabulary. A decision model scores only the answers you declared, in a single pass, and is trained so its probabilities can be used as thresholds.

**What is OpenAI's Decisions API?**
A limited-preview API announced at DevDay 2026 that uses a version of GPT-6 Luna to answer user-defined questions with finite predefined answers, from text or image context. It is reported at about 150 ms per decision.

**Is Laya an alternative to OpenAI's Decisions API?**
For text-only decisions, yes. Laya is open (Apache-2.0), answers choice, score and yes/no questions, and runs locally or in a browser. It doesn't take images, and it is weaker on very long inputs and large label sets.

**Do decision models hallucinate?**
They can't return an answer outside the list you define. They can still choose the wrong listed answer with high confidence, so validate thresholds on your own data and keep a human in the loop for costly actions.

**Will decision models replace LLMs in agents?**
No. They take over the narrow, repeated judgments (route, gate, score, choose the next step) while the language model keeps planning, reasoning and writing.

---

## Disclaimer

This article is an independent technical overview written for educational purposes. Details of OpenAI's Decisions API come from OpenAI's DevDay 2026 recap and third-party launch coverage of a **limited-preview** product. The schema, capabilities, limits and performance figures may change and should be checked against OpenAI's official documentation. The ~150 ms and ~10× figures are reported launch claims, not independent measurements. Jev details are drawn from public documentation and third-party write-ups. Laya and Jev benchmark figures are quoted from Laya's model card, where the Jev numbers are third-party published with different samples and prompts. Latency figures from different sources were measured on different hardware and aren't directly comparable.

The layaMOE results come from a single run on a small, author-labeled evaluation set (108 cases, 312 answers), with expert heads trained on synthetic data. They are experimental observations, not production guarantees. layaMOE is an unofficial project, not affiliated with or endorsed by OpenAI, TypeSafe AI, ConvAI Innovations or any other company named here. All product names and trademarks belong to their respective owners. No model named here should be the only safeguard for consequential decisions.

**Sources:**
[OpenAI DevDay 2026 recap](https://openai.com/index/devday-2026-recap/) ·
[The Decoder: OpenAI expands Codex and its API at DevDay](https://the-decoder.com/openai-expands-codex-and-its-api-at-devday-with-security-scans-a-decisions-api-and-ultrafast/) ·
[FourWeekMBA: OpenAI Decisions API limits the output space itself](https://fourweekmba.com/ai-openai-decisions-api-finite-output-space/) ·
[What Is Decisions API? (Hugging Face community article)](https://huggingface.co/blog/sora-2/what-is-openai-decisions-api-a-practical-guide) ·
[Runware: Jev, Laya and the emerging role of decision models](https://runware.ai/blog/jev-laya-and-the-emerging-role-of-decision-models) ·
[laya-typed-decisions model card](https://huggingface.co/convaiinnovations/laya-typed-decisions) ·
[What Is Jev? (OpenRouter)](https://openrouter.ai/blog/insights/what-is-jev/) ·
[awesome-system-one](https://github.com/yanng981/awesome-system-one) ·
[Kev](https://github.com/jaredpalmer/kev) ·
[Von](https://github.com/wfzyx/von) ·
[layaMOE](https://github.com/vishalmysore/layaMOE)
