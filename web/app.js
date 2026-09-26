import * as ort from "./vendor/ort.min.mjs";
import { Tokenizer } from "./vendor/tokenizers.min.mjs";
import { LayaMoE } from "./laya-core.js";

const $ = (id) => document.getElementById(id);
// Model files live on Hugging Face (too big for GitHub Pages). Override with ?modelBase=... or site-config.json.
const DEFAULT_BASE = "https://huggingface.co/VishalMysore/laya-moe-web/resolve/main/";
let BASE = DEFAULT_BASE;
let MANIFEST = null;
let moe = null;
let lastResult = null;

const PRESETS = {
  "Agent guardrail: drop a production table": {
    state: "Agent plan: run `DROP TABLE invoices` on the production database. No backup has been taken and nobody has reviewed the command.",
    questions: {
      destructive: { type: "noul", instructions: "The action is destructive and cannot be undone" },
      needs_human: { type: "noul", instructions: "A human should approve this action before it runs" },
      risk: { type: "score", instructions: "How risky is this action?", criteria: ["Low", "Medium", "High", "Critical"] },
    },
  },
  "Agent guardrail: dry run": {
    state: "Agent plan: run `rm -rf /srv/uploads` on the staging server with `--dry-run`, which only prints what would be removed.",
    questions: {
      destructive: { type: "noul", instructions: "The action is destructive and cannot be undone" },
      safe_without_approval: { type: "noul", instructions: "It is safe to run this action without a human approving it first" },
      risk: { type: "score", instructions: "How risky is this action?", criteria: ["Low", "Medium", "High", "Critical"] },
    },
  },
  "Comment moderation": {
    state: "Honestly the author of this post is a clueless clown, no idea why anyone reads this blog.",
    questions: {
      verdict: { type: "choice", instructions: "What should happen to this comment?", criteria: { allow: "Fine to publish", review: "Needs a human moderator", remove: "Breaks the rules and should be removed" } },
      toxicity: { type: "score", instructions: "How toxic is the language?", criteria: ["None", "Mild", "Strong", "Severe"] },
      spam: { type: "noul", instructions: "The comment is spam or an advertisement" },
    },
  },
  "Support ticket": {
    state: { ticket: { subject: "Invoices page broken", text: "The invoices page throws an error since this morning. Our month-end close is today and finance is blocked." } },
    questions: {
      team: { type: "choice", instructions: "Which team should handle this?", criteria: { bug: "Something is broken", how_to: "A usage question", feature_request: "A request for a new capability", account_access: "Login, password or account access" } },
      urgency: { type: "score", instructions: "How urgent is this?", criteria: ["Can wait", "This week", "Today", "Right now"] },
      angry: { type: "noul", instructions: "The customer sounds angry" },
    },
  },
  "Delivery exception": {
    state: "Tracking has not moved for 12 days and the customer has written three times asking where the order is.",
    questions: {
      action: { type: "choice", instructions: "What should support do?", criteria: { reship: "Send a replacement", notify: "Tell the customer about the delay or outcome", fix_address: "Contact the customer to fix the delivery details", wait: "Wait, no action needed" } },
      urgency: { type: "score", instructions: "How urgent is this?", criteria: ["Not urgent", "Soon", "Urgent", "Immediate"] },
      upset: { type: "noul", instructions: "The customer is upset" },
    },
  },
  "Work email": {
    state: "Hi, my laptop won't connect to the VPN since the update. Who handles IT issues? Thanks, Grace",
    questions: {
      action: { type: "choice", instructions: "What should the recipient do with this email?", criteria: { reply_now: "Reply today", reply_later: "Reply within the week", archive: "No reply needed", delegate: "Forward it to someone else" } },
      needs_reply: { type: "noul", instructions: "The sender expects a reply" },
    },
  },
  "Out of domain: sales lead (general head)": {
    state: "Hi, I'm the VP of Engineering at a 400-person logistics company. We have budget approved and want a demo next week with our security team.",
    questions: {
      lead_quality: { type: "score", instructions: "How qualified is this sales lead?", criteria: ["Not a fit", "Weak interest", "Some interest", "Strong buying signals"] },
      next_step: { type: "choice", instructions: "What should sales do next?", criteria: { book_demo: "Schedule a demo", nurture: "Add to a nurture campaign", ignore: "No action needed" } },
    },
  },
};

function setStatus(msg, cls = "") { const s = $("status"); s.textContent = msg; s.className = "status " + cls; }
function setProgress(f) { $("progress").style.display = f == null ? "none" : "block"; if (f != null) $("progressBar").style.width = (f * 100).toFixed(1) + "%"; }
function badge(text, ok = false) { const b = document.createElement("span"); b.className = "badge" + (ok ? " ok" : ""); b.textContent = text; $("badges").appendChild(b); }

async function fetchBytes(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length; onProgress?.(got, total);
  }
  const out = new Uint8Array(got); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

// Weight files are stored as <= 24 MiB parts; parts are cached in Cache Storage keyed by the file hash.
let cachedParts = 0;
async function fetchParts(entry, onProgress) {
  let cache = null;
  try { cache = await caches.open("laya-moe-" + entry.sha256.slice(0, 16)); } catch { /* private mode: just download */ }
  const out = new Uint8Array(entry.size); let off = 0;
  for (const part of entry.parts) {
    const url = BASE + part;
    let bytes = null;
    try { const hit = cache && await cache.match(url); if (hit) { bytes = new Uint8Array(await hit.arrayBuffer()); cachedParts++; } } catch {}
    if (!bytes) {
      bytes = await fetchBytes(url, (got) => onProgress(off + got));
      try { if (cache) await cache.put(url, new Response(bytes)); } catch { /* quota exceeded: fine */ }
    }
    if (off + bytes.length > entry.size) throw new Error("weights are larger than the manifest says");
    out.set(bytes, off); off += bytes.length; onProgress(off);
  }
  if (off !== entry.size) throw new Error(`weights incomplete: got ${off} of ${entry.size} bytes`);
  return out;
}

async function createSession(entry, onProgress) {
  const graph = await fetchBytes(BASE + entry.onnx);
  const data = await fetchParts(entry.data, onProgress);
  return ort.InferenceSession.create(graph, { executionProviders: ["wasm"], graphOptimizationLevel: "all", externalData: [{ path: entry.data.name, data }] });
}

async function loadModel() {
  $("loadBtn").disabled = true; $("runBtn").disabled = true; $("badges").innerHTML = ""; moe = null; cachedParts = 0;
  try {
    ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    setStatus("Loading tokenizer and config…"); setProgress(0);
    const [tj, tc, cfg] = await Promise.all(["tokenizer.json", "tokenizer_config.json", "rl_agent_config.json"].map((f) =>
      fetch(BASE + f).then((r) => { if (!r.ok) throw new Error(f + ": HTTP " + r.status); return r.json(); })));
    const tokenizer = new Tokenizer(tj, tc);

    const all = [["encoder", MANIFEST.encoder], ...Object.entries(MANIFEST.heads)];
    const totalBytes = all.reduce((a, [, e]) => a + e.data.size, 0);
    let doneBytes = 0;
    const t0 = performance.now();
    const sessions = {};
    for (const [name, entry] of all) {
      const base = doneBytes;
      sessions[name] = await createSession(entry, (got) => {
        setProgress((base + got) / totalBytes);
        setStatus(`Downloading ${name === "encoder" ? "shared encoder" : "head: " + name} (${((base + got) / 1048576).toFixed(0)} / ${(totalBytes / 1048576).toFixed(0)} MB)`);
      });
      doneBytes += entry.data.size;
    }
    const heads = Object.fromEntries(Object.entries(MANIFEST.heads).map(([n, m]) => [n, { session: sessions[n], meta: m }]));
    moe = new LayaMoE(ort, sessions.encoder, heads, tokenizer, cfg, MANIFEST.router);
    setStatus("Warming up…"); setProgress(null);
    await moe.systemOne("warm up", { w: { type: "noul", instructions: "This is a warm-up call" } });
    const parts = all.reduce((a, [, e]) => a + e.data.parts.length, 0);
    setStatus(`Ready in ${((performance.now() - t0) / 1000).toFixed(1)} s${cachedParts ? ` (${cachedParts} of ${parts} parts from browser cache)` : ""}.`);
    badge("WASM" + (ort.env.wasm.numThreads > 1 ? ` · ${ort.env.wasm.numThreads} threads` : " · 1 thread"), true);
    badge(`${Object.keys(heads).length} heads: ${Object.keys(heads).join(", ")}`);
    badge("cross-origin isolated: " + (self.crossOriginIsolated ? "yes" : "no"));
    $("runBtn").disabled = false;
  } catch (e) {
    console.error(e); setStatus("Could not load the model: " + (e?.message || e), "warn");
  } finally { setProgress(null); $("loadBtn").disabled = false; }
}

const pct = (x) => (x * 100).toFixed(1) + "%";
function barRow(name, p, top, cls = "") {
  const d = document.createElement("div"); d.className = "bar" + (top ? " top" : "") + (cls ? " " + cls : "");
  d.innerHTML = `<span class="name"></span><span class="track"><span class="fill" style="display:block;width:${(p * 100).toFixed(1)}%"></span></span><span class="val">${pct(p)}</span>`;
  d.querySelector(".name").textContent = name; d.querySelector(".name").title = name;
  return d;
}
const headline = (a) => a.type === "choice" ? a.choice : a.type === "score" ? `score ${a.score.toFixed(2)}` : `P(true) ${pct(a.noul)}`;
function bars(box, a, cls = "") {
  if (a.type === "choice") { const mx = Math.max(...Object.values(a.probabilities)); Object.entries(a.probabilities).forEach(([k, p]) => box.appendChild(barRow(k, p, p === mx, cls))); }
  else if (a.type === "score") { const mx = Math.max(...Object.values(a.probabilities)); Object.entries(a.probabilities).forEach(([i, p]) => box.appendChild(barRow(`${i}: ${a.legend[i]}`, p, p === mx, cls))); }
  else { box.appendChild(barRow("yes", a.noul, a.noul >= 0.5, cls)); box.appendChild(barRow("no", 1 - a.noul, a.noul < 0.5, cls)); }
}

function render(res) {
  lastResult = res;
  $("resultCard").style.display = "block";
  $("sExpert").textContent = res.expert;
  $("sEnc").textContent = res.latency_ms.encoder.toFixed(0);
  $("sHeads").textContent = res.latency_ms.heads.toFixed(0);
  $("sTokens").textContent = res.usage.input_tokens;
  $("raw").textContent = JSON.stringify(res, null, 2);

  const r = res.routing, rb = $("routing"); rb.innerHTML = "";
  const map = MANIFEST.router.experts || {};
  const mx = Math.max(...Object.values(r.probabilities));
  Object.entries(r.probabilities).sort((a, b) => b[1] - a[1]).slice(0, 5)
    .forEach(([k, p]) => rb.appendChild(barRow(`${k} → ${map[k] || "general"}`, p, p === mx)));
  const note = document.createElement("div"); note.className = "meta";
  note.textContent = r.overridden
    ? `You forced the "${res.expert}" head. The router would have picked "${r.router_pick}".`
    : r.router_pick === "general"
      ? `The router read this as "${r.top}", which has no expert (or is below the ${r.threshold} threshold), so the general head answered.`
      : `The router read this as "${r.top}", so the "${r.router_pick}" expert answered.`;
  rb.appendChild(note);

  const thr = Number($("thresh").value);
  const box = $("answers"); box.innerHTML = "";
  for (const [qid, a] of Object.entries(res.answers)) {
    const q = document.createElement("div"); q.className = "q";
    const g = res.general_answers?.[qid];
    const head = document.createElement("div"); head.className = "qhead";
    head.innerHTML = `<span><span class="qname"></span> <span class="qtype">${a.type}</span></span><span><b class="hl"></b></span>`;
    head.querySelector(".qname").textContent = qid;
    head.querySelector(".hl").textContent = headline(a) + (g ? `  (general: ${headline(g)})` : "");
    q.appendChild(head);
    if (g) {
      const cols = document.createElement("div"); cols.className = "cols tight";
      const c1 = document.createElement("div"), c2 = document.createElement("div");
      c1.innerHTML = `<div class="colhead">${res.expert} expert</div>`; c2.innerHTML = `<div class="colhead">general head</div>`;
      bars(c1, a); bars(c2, g, "muted");
      cols.append(c1, c2); q.appendChild(cols);
    } else bars(q, a);
    const m = document.createElement("div"); m.className = "meta"; m.textContent = `confidence ${a.confidence.toFixed(3)}`; q.appendChild(m);
    const dec = document.createElement("div"); dec.className = "decision";
    dec.innerHTML = a.confidence >= thr ? `Your code would <b class="auto">act automatically</b> (confidence ≥ ${thr.toFixed(2)})` : `Your code would <b class="human">send to a person</b> (confidence below ${thr.toFixed(2)})`;
    q.appendChild(dec);
    box.appendChild(q);
  }
}

async function run() {
  if (!moe) return;
  let state, questions;
  const raw = $("state").value.trim();
  try { state = raw.startsWith("{") || raw.startsWith("[") ? JSON.parse(raw) : $("state").value; } catch { state = $("state").value; }
  try { questions = JSON.parse($("questions").value); } catch (e) { setStatus("Questions is not valid JSON: " + e.message, "warn"); return; }
  $("runBtn").disabled = true;
  try { render(await moe.systemOne(state, questions, { expert: $("expert").value, compare: $("compare").checked })); setStatus("Done."); }
  catch (e) { console.error(e); setStatus("Run failed: " + (e?.message || e), "warn"); }
  finally { $("runBtn").disabled = false; }
}

async function init() {
  let dir = new URLSearchParams(location.search).get("modelBase");
  if (!dir) { try { const c = await fetch("./site-config.json"); if (c.ok) dir = (await c.json()).modelBase; } catch {} }
  BASE = (dir || DEFAULT_BASE).replace(/\/?$/, "/");
  try {
    const r = await fetch(BASE + "manifest.json");
    if (!r.ok) throw new Error("HTTP " + r.status);
    MANIFEST = await r.json();
  } catch (e) { setStatus(`Could not read ${BASE}manifest.json (${e.message}). Have the web files been uploaded?`, "warn"); return; }
  const sel = $("expert"); sel.innerHTML = ""; sel.add(new Option("Auto (router decides)", "auto"));
  for (const [n, h] of Object.entries(MANIFEST.heads)) sel.add(new Option(n === "general" ? "general (original head)" : `${n} (${h.domains.join(", ")})`, n));
  const mb = (e) => e.data.size / 1048576;
  const total = mb(MANIFEST.encoder) + Object.values(MANIFEST.heads).reduce((a, h) => a + mb(h), 0);
  $("sizes").textContent = `Shared encoder ~${mb(MANIFEST.encoder).toFixed(0)} MB + ${Object.keys(MANIFEST.heads).length} heads ~${mb(Object.values(MANIFEST.heads)[0]).toFixed(0)} MB each = ~${total.toFixed(0)} MB, downloaded once and cached.`;
  setStatus("Not loaded yet.");
  $("loadBtn").disabled = false;
}

for (const k of Object.keys(PRESETS)) $("preset").add(new Option(k, k));
function applyPreset() {
  const p = PRESETS[$("preset").value];
  $("state").value = typeof p.state === "string" ? p.state : JSON.stringify(p.state, null, 2);
  $("questions").value = JSON.stringify(p.questions, null, 2);
}
$("preset").addEventListener("change", applyPreset); applyPreset();
$("thresh").addEventListener("input", () => { $("threshVal").textContent = Number($("thresh").value).toFixed(2); if (lastResult) render(lastResult); });
$("loadBtn").addEventListener("click", loadModel);
$("runBtn").addEventListener("click", run);
document.addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") run(); });
$("loadBtn").disabled = true;
init();
window.__moe = { loadModel, run, get ready() { return !!moe; }, get instance() { return moe; } };
