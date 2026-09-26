"""Laya MoE runtime: one shared frozen encoder, several expert heads, a router, and a general fallback.

    from laya_moe.moe import MoEAgent
    moe = MoEAgent(experts=["checkpoints/safety", "checkpoints/customer_ops"])
    out = moe.system_one(state, questions)          # routes, then answers with the chosen head
    out["expert"], out["routing"]                     # which head answered and why

Routing is sequence-level (once per request, not per token): the base laya-typed-decisions model
answers one choice question, "what kind of text is this?", over fine-grained kinds (agent action,
public comment, support ticket, ...). If the top kind maps to a loaded expert and its probability is
>= threshold, that expert's head answers; otherwise the untouched base head does.
"""
import numpy as np
import torch

from laya.common import QTYPES, confidence_from_probs, temp_bucket

from .core import BASE_MODEL, encode, load_base, load_expert, make_item, to_internal

# Router: the base model picks what *kind* of text this is from fine-grained options (much easier for it
# than broad "which expert" options); each kind maps to an expert, anything unmapped goes to "general".
ROUTER_OPTIONS = {
    "agent_action": "A plan or command an AI agent wants to run on software, data or infrastructure",
    "public_comment": "A comment posted by a reader on a blog, forum or social media",
    "support_ticket": "A customer asking a software company for help with its product",
    "delivery_issue": "A problem with shipping or delivering a parcel",
    "work_email": "An email between colleagues at work",
    "it_alert": "A monitoring alert or report about an IT system incident",
    "patient_message": "A message from a patient to a doctor or clinic",
    "product_review": "A customer review of a product they bought",
    "sales_inquiry": "A potential customer contacting a sales team",
    "other": "Something else",
}
ROUTER_EXPERTS = {"agent_action": "safety", "public_comment": "safety",
                  "support_ticket": "customer_ops", "delivery_issue": "customer_ops", "work_email": "customer_ops"}
ROUTER_QUESTION = {"type": "choice", "instructions": "What kind of text is this?", "criteria": ROUTER_OPTIONS}


class MoEAgent:
    def __init__(self, experts=(), base_model=BASE_MODEL, threshold=0.3):
        self.base = load_base(base_model)
        self.tok, self.cfg = self.base.tok, self.base.cfg
        self.threshold = threshold
        self.experts = {}
        for path in experts:
            head, meta = load_expert(path, self.base.model)
            self.experts[meta["expert"]] = (head, meta)
        self.router_q = ROUTER_QUESTION

    # -- routing -------------------------------------------------------------------------------
    @torch.no_grad()
    def route(self, state):
        a = self.base.system_one(state, {"route": self.router_q})["answers"]["route"]
        probs = a["probabilities"]
        top = max(probs, key=probs.get)
        exp = ROUTER_EXPERTS.get(top)
        chosen = exp if (exp in self.experts and probs[top] >= self.threshold) else "general"
        return chosen, {"probabilities": probs, "top": top, "confidence": a["confidence"]}

    # -- answering -----------------------------------------------------------------------------
    @torch.no_grad()
    def answer_with(self, name, state, questions):
        """Answer with a named head ("general" = the base checkpoint's own head)."""
        ids = list(questions)
        items = [make_item(self.tok, self.cfg, state, questions[q]) for q in ids]
        h, att, mpos, mmask, qt = encode(self.base.model.encoder, items, self.tok.pad_token_id)
        if name == "general":
            m = self.base.model
            x = h + m.type_emb(qt)[:, None, :]
            for layer in m.head.layers:
                x = layer(x, src_key_padding_mask=~att.bool())
            idx = mpos.clamp(min=0)[:, :, None].expand(-1, -1, x.size(-1))
            logits = m.scorer(torch.gather(x, 1, idx)).squeeze(-1).float().masked_fill(~mmask, -1e4)
            temp, temp_opts = self.cfg.get("temperature", [1, 1, 1]), self.cfg.get("temperature_by_options", {})
        else:
            head, meta = self.experts[name]
            logits = head(h, att, mpos, mmask, qt)
            temp, temp_opts = meta["temperature"], meta["temperature_by_options"]
        logits = logits.numpy()
        answers = {}
        for r, qid in enumerate(ids):
            q = to_internal(questions[qid])
            k = len(items[r]["markers"])
            qtype = QTYPES[q["t"]]
            T = temp_opts.get(temp_bucket(qtype, k), temp[qtype])
            z = logits[r, :k] / max(1e-3, float(T))
            p = np.exp(z - z.max())
            p /= p.sum()
            conf = round(confidence_from_probs(p, k), 4)
            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {"type": "choice", "choice": keys[int(p.argmax())],
                                "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)}, "confidence": conf}
            elif q["t"] == "score":
                answers[qid] = {"type": "score", "score": round(float((np.arange(k) * p).sum()), 4),
                                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)}, "confidence": conf}
            else:
                answers[qid] = {"type": "noul", "noul": round(float(p[1]), 4),
                                "confidence": round(max(float(p[1]), 1 - float(p[1])), 4)}
        return answers

    def system_one(self, state, questions, expert=None):
        routing = None
        if expert is None:
            expert, routing = self.route(state)
        return {"expert": expert, "routing": routing, "answers": self.answer_with(expert, state, questions)}
