"""Generate synthetic training texts for the trained router's "general" kinds.

The expert domains already have training data in data/train/<expert>/. The router also has to learn
what does *not* belong to an expert, so this script writes template texts for the kinds that stay on
the general head: IT alerts, patient messages, product reviews, sales inquiries and "other" (news,
recipes, meeting notes, ...). Each text is paired with a few generic typed questions so the router
sees realistic (text, question) inputs.

Like gen_train_data.py, any text that shares a word 5-gram with a data/eval case is dropped.

    python scripts/gen_router_data.py                # writes data/train/router/<kind>.jsonl
"""
import argparse, glob, json, os, random, re

R = random.Random()
pick = lambda xs: R.choice(xs)
maybe = lambda p: R.random() < p


def ngrams(text, n=5):
    w = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


SYSTEMS = ["payments gateway", "Kafka cluster", "Redis cache", "Kubernetes node pool", "VPN concentrator", "mail relay",
           "DNS resolver", "data warehouse", "CI runners", "object storage", "load balancer", "LDAP server", "backup job"]
SYMPTOMS = ["disk usage at {n}%", "p99 latency above {m} ms", "error rate at {n}%", "memory at {n}% and climbing",
            "certificate expires in {d} days", "replication lag of {m} seconds", "{n} pods in CrashLoopBackOff",
            "queue depth over {m}k messages", "health check failing on {d} of 6 hosts", "CPU steady at {n}%"]


def it_alert():
    s = pick(SYMPTOMS).format(n=R.randint(70, 99), m=R.randint(2, 90) * 10, d=R.randint(1, 5))
    head = pick(["ALERT", "[PagerDuty]", "Monitoring:", "Incident report:", "Grafana alert -", "Nagios WARNING:", "Datadog monitor triggered:"])
    tail = pick(["", " No customer reports so far.", " Several users report slowness.", " Auto-remediation failed.",
                 " On-call has been paged.", " Started after the 14:00 deploy.", " Affects the EU region only."])
    return f"{head} {pick(SYSTEMS)} {s}.{tail}"


def patient_message():
    who = pick(["I", "My son", "My mother", "My husband", "My daughter", "I"])
    issue = pick(["have had a sore throat for {d} days", "need a refill of my blood pressure tablets",
                  "would like to move my appointment on {day}", "got my lab results and have a question about them",
                  "have a rash on my arm since {day}", "keep waking up with headaches", "twisted my ankle playing football",
                  "need a sick note for work", "have a fever of 38.{n} since yesterday", "want to ask about the flu vaccine",
                  "am feeling dizzy after starting the new medication", "need a referral to a physiotherapist"]).format(
        d=R.randint(2, 9), n=R.randint(1, 9), day=pick(["Monday", "Tuesday", "Thursday", "the 12th", "next week"]))
    if who != "I":
        issue = issue.replace("have ", "has ").replace("need ", "needs ").replace("would ", "would ").replace("keep ", "keeps ") \
            .replace("got my", "got their").replace("my arm", "their arm").replace("am feeling", "is feeling").replace("want ", "wants ")
    greet = pick(["", "Hello doctor, ", "Hi, ", "Dear clinic, ", "Good morning, "])
    end = pick(["", " Thank you.", " Can someone call me back?", " Is this something to worry about?", " What should I do?"])
    if greet and who != "I":
        who = who[0].lower() + who[1:]
    return f"{greet}{who} {issue}.{end}"


PRODUCTS = ["wireless earbuds", "standing desk", "air fryer", "hiking backpack", "robot vacuum", "espresso machine",
            "office chair", "phone case", "electric toothbrush", "4K monitor", "yoga mat", "smart watch", "rice cooker"]


def product_review():
    p = pick(PRODUCTS)
    stars = R.randint(1, 5)
    body = {1: ["Broke after two weeks.", "Complete waste of money.", "Nothing like the photos."],
            2: ["Disappointing build quality.", "Works, but the battery is poor.", "Too noisy for daily use."],
            3: ["It's okay for the price.", "Does the job, nothing special.", "Good design, average performance."],
            4: ["Really happy with it overall.", "Solid and easy to set up.", "Great value, minor quirks."],
            5: ["Best purchase this year!", "Exceeded my expectations.", "Absolutely love it, highly recommend."]}[stars]
    extra = pick(["", " Delivery was quick.", " The instructions were confusing.", " My partner uses it every day.",
                  " Would buy again.", " Customer service was helpful.", " Returned it for a refund."])
    return pick([f"{stars}/5 - {p}: ", f"Review of the {p}. ", f"Rated {stars} stars. ", f"{p.title()}: "]) + pick(body) + extra


def sales_inquiry():
    size = pick(["a 10-person startup", "a team of 40", "a 300-seat company", "an agency with 12 staff", "a school district",
                 "a hospital network", "a solo consultant", "an enterprise with 5,000 employees"])
    ask = pick(["Could you send pricing for the business plan?", "We'd like a demo next week.",
                "Do you offer volume discounts?", "Is there an annual contract option?", "Can we trial it for a month?",
                "What integrations do you support?", "Who can I talk to about a pilot?", "Do you have a reseller programme?"])
    return pick(["Hi, ", "Hello sales team, ", "Hey, ", ""]) + f"we're {size} looking at your product. {ask}" + \
        pick(["", " Budget is approved for this quarter.", " Just exploring options for now.", " We're comparing three vendors."])


OTHER = [
    lambda: f"Recipe: {pick(['lentil soup', 'banana bread', 'chicken curry', 'pancakes', 'mushroom risotto'])}. {pick(['Serves 4.', 'Ready in 30 minutes.', 'Preheat the oven to 180C.'])}",
    lambda: f"Meeting notes {pick(['Monday', 'Q3 planning', 'design sync', 'retro'])}: {pick(['agreed to ship the beta', 'budget stays flat', 'hiring freeze until March', 'next review in two weeks'])}.",
    lambda: f"News: {pick(['the city council approved a new bike lane', 'a heatwave is expected this weekend', 'the local team won the cup final', 'the museum reopens after renovation'])}.",
    lambda: f"Trivia: {pick(['what is the capital of Australia?', 'how many moons does Mars have?', 'who wrote Pride and Prejudice?', 'what is the boiling point of water in Fahrenheit?'])}",
    lambda: f"Travel plan: {pick(['fly to Lisbon on Friday', 'train to Montreal at 9am', 'road trip along the coast', 'two nights in Kyoto'])}, {pick(['hotel booked', 'need a rental car', 'pack a raincoat', 'check passport expiry'])}.",
    lambda: f"Code review: {pick(['rename this variable for clarity', 'add a unit test for the null case', 'this loop can be a list comprehension', 'nice refactor, approved'])}.",
    lambda: f"Poem draft: {pick(['the river hums beneath the bridge', 'autumn leaves in quiet rows', 'a lantern in the harbour fog'])}.",
    lambda: f"Shopping list: {', '.join(R.sample(['milk', 'eggs', 'bread', 'apples', 'coffee', 'rice', 'soap', 'tomatoes'], 4))}.",
    lambda: f"Weather update: {pick(['sunny with a high of 24', 'light rain in the afternoon', 'snow expected overnight', 'windy along the coast'])}.",
    lambda: f"Book summary: {pick(['a detective returns to her hometown', 'two siblings inherit a failing vineyard', 'an astronaut is stranded on Mars'])}.",
]

GENERIC_Q = [
    {"type": "choice", "instructions": "What is the main topic?", "criteria": {"work": "Work related", "personal": "Personal", "other": "Other"}},
    {"type": "score", "instructions": "How urgent is this?", "criteria": ["Not urgent", "Soon", "Today", "Immediately"]},
    {"type": "noul", "instructions": "This needs a reply from a person"},
    {"type": "score", "instructions": "How positive is the tone?", "criteria": ["Negative", "Neutral", "Positive"]},
    {"type": "noul", "instructions": "The text describes a problem"},
    {"type": "choice", "instructions": "Who should handle this?", "criteria": {"team_a": "Operations", "team_b": "Sales", "team_c": "Nobody"}},
]

KINDS = {"it_alert": it_alert, "patient_message": patient_message, "product_review": product_review,
         "sales_inquiry": sales_inquiry, "other": lambda: pick(OTHER)()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-kind", type=int, default=400)
    ap.add_argument("--seed", type=int, default=4321)
    here = os.path.dirname(__file__)
    ap.add_argument("--out", default=os.path.join(here, "..", "data", "train", "router"))
    ap.add_argument("--eval", default=os.path.join(here, "..", "data", "eval"))
    args = ap.parse_args()
    R.seed(args.seed)
    eval_grams = set()
    for p in glob.glob(os.path.join(args.eval, "*.json")):
        if p.endswith("index.json"):
            continue
        for c in json.load(open(p, encoding="utf-8"))["cases"]:
            eval_grams |= ngrams(c["state"] if isinstance(c["state"], str) else json.dumps(c["state"]))
    os.makedirs(args.out, exist_ok=True)
    for kind, gen in KINDS.items():
        seen, rows, dropped, tries = set(), [], 0, 0
        while len(rows) < args.per_kind and tries < args.per_kind * 200:
            tries += 1
            t = gen()
            if t in seen:
                continue
            seen.add(t)
            if ngrams(t) & eval_grams:
                dropped += 1
                continue
            qs = {f"q{i}": q for i, q in enumerate(R.sample(GENERIC_Q, 2))}
            rows.append({"kind": kind, "state": t, "questions": qs})
        with open(os.path.join(args.out, kind + ".jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{kind:16} {len(rows):4d} texts (dropped {dropped} too close to eval)")


if __name__ == "__main__":
    main()
