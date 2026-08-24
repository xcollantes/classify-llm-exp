"""HTML assembly for the experiment report.

Charts are rendered by report.py; this module turns the validated results plus
those PNGs into the single self-contained results/report.html.
"""

import base64
from pathlib import Path

from experiment import EMBED_MODEL_ID, GEMINI_LLM_MODEL_ID, OPENAI_LLM_MODEL_ID
from models import ExperimentResults, SystemMetrics

CHARTS: Path = Path("results") / "charts"

ORDER: list[str] = [
    GEMINI_LLM_MODEL_ID,
    OPENAI_LLM_MODEL_ID,
    "embed-CLASSIFICATION",
    "embed-SEMANTIC_SIMILARITY",
]
SHORT: dict[str, str] = {
    GEMINI_LLM_MODEL_ID: "Gemini 3.5 Flash-Lite\n(zero-shot)",
    OPENAI_LLM_MODEL_ID: "GPT-5.4 nano\n(zero-shot)",
    "embed-CLASSIFICATION": "gemini-embedding-001\nCLASSIFICATION + LR",
    "embed-SEMANTIC_SIMILARITY": "gemini-embedding-001\nSEMANTIC_SIMILARITY + LR",
}


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def figure(name: str, caption: str) -> str:
    """<figure> with light/dark PNG pair, swapped by CSS."""
    light, dark = CHARTS / f"{name}_light.png", CHARTS / f"{name}_dark.png"
    return f"""
<figure>
  <img class="light-only" src="data:image/png;base64,{b64(light)}" alt="{caption}">
  <img class="dark-only" src="data:image/png;base64,{b64(dark)}" alt="{caption}">
  <figcaption>{caption}</figcaption>
</figure>"""


def candidates_table(m: ExperimentResults) -> str:
    """The three candidates side by side: provenance, price, and how they did.

    The embedding row reports the headline CLASSIFICATION arm; its
    SEMANTIC_SIMILARITY twin is the control, covered in section 4.
    """
    sys, price = m.systems, m.config.pricing_usd_per_1m_tokens
    rows = ""
    for model_id, arm, role in (
        (GEMINI_LLM_MODEL_ID, GEMINI_LLM_MODEL_ID, "LLM · zero-shot"),
        (OPENAI_LLM_MODEL_ID, OPENAI_LLM_MODEL_ID, "LLM · zero-shot"),
        (EMBED_MODEL_ID, "embed-CLASSIFICATION", "Embedding + LR"),
    ):
        p, s = price[model_id], sys[arm]
        out = f"${p.output_per_1m:.2f}" if p.output_per_1m else "—"
        rows += (
            f"<tr><td><b>{p.name}</b><br>"
            f"<span class='ci'><code>{p.model}</code></span></td>"
            f"<td>{p.maker}</td>"
            f"<td>{p.released}</td>"
            f"<td>{role}</td>"
            f"<td class='num'>${p.input_per_1m:.2f} / {out}</td>"
            f"<td class='num'><b>{s.accuracy:.1%}</b></td>"
            f"<td class='num'>{s.macro_f1:.3f}</td>"
            f"<td class='num'>${s.cost_usd_per_1k_docs:.3f}</td></tr>"
        )
    return rows


def metrics_table(m: ExperimentResults) -> str:
    """Render the results table body, one <tr> per arm."""
    sys: dict[str, SystemMetrics] = m.systems
    rows = ""
    for n in ORDER:
        s = sys[n]
        lo, hi = s.accuracy_ci95
        rows += (
            f"<tr><td>{SHORT[n].replace(chr(10), ' ')}</td>"
            f"<td>{s.kind}</td>"
            f"<td>{s.labeled_examples}</td>"
            f"<td class='num'><b>{s.accuracy:.3f}</b> "
            f"<span class='ci'>[{lo:.3f}–{hi:.3f}]</span></td>"
            f"<td class='num'>{s.macro_f1:.3f}</td>"
            f"<td class='num'>{s.latency_mean_s*1000:.0f} ms</td>"
            f"<td class='num'>${s.cost_usd_per_1k_docs:.3f}</td>"
            f"<td class='num'>{s.unparsed}</td></tr>")
    return rows


PIPELINE_SVG: str = """
<svg viewBox="0 0 900 250" role="img" aria-label="Three classification pipelines"
     xmlns="http://www.w3.org/2000/svg">
  <style>
    .bx { fill: var(--panel); stroke: var(--grid); stroke-width: 1.5; rx: 8; }
    .lb { fill: var(--text); font: 600 13px system-ui, sans-serif; }
    .sb { fill: var(--muted); font: 11px system-ui, sans-serif; }
    .ar { stroke: var(--muted); stroke-width: 1.5; fill: none; marker-end: url(#a); }
  </style>
  <defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6"
      markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="var(--muted)"/>
  </marker></defs>

  <rect class="bx" x="8" y="95" width="130" height="58"/>
  <text class="lb" x="73" y="118" text-anchor="middle">400 test posts</text>
  <text class="sb" x="73" y="136" text-anchor="middle">20 Newsgroups, 4 classes</text>

  <path class="ar" d="M142 108 H 300"/>
  <path class="ar" d="M142 124 H 300"/>
  <path class="ar" d="M142 140 H 300"/>

  <rect class="bx" x="304" y="18" width="230" height="56" style="stroke:var(--s1)"/>
  <text class="lb" x="419" y="40" text-anchor="middle">Gemini 3.5 Flash-Lite</text>
  <text class="sb" x="419" y="58" text-anchor="middle">zero-shot prompt, temp 0</text>

  <rect class="bx" x="304" y="96" width="230" height="56" style="stroke:var(--s2)"/>
  <text class="lb" x="419" y="118" text-anchor="middle">GPT-5.4 nano</text>
  <text class="sb" x="419" y="136" text-anchor="middle">same prompt, temp 0</text>

  <rect class="bx" x="304" y="174" width="230" height="62" style="stroke:var(--s3)"/>
  <text class="lb" x="419" y="196" text-anchor="middle">gemini-embedding-001</text>
  <text class="sb" x="419" y="213" text-anchor="middle">task_type = CLASSIFICATION</text>
  <text class="sb" x="419" y="229" text-anchor="middle">→ logistic regression (800 labelled)</text>

  <path class="ar" d="M538 46 H 700"/>
  <path class="ar" d="M538 124 H 700"/>
  <path class="ar" d="M538 205 H 700"/>

  <rect class="bx" x="704" y="95" width="180" height="58"/>
  <text class="lb" x="794" y="118" text-anchor="middle">one label per post</text>
  <text class="sb" x="794" y="136" text-anchor="middle">accuracy · F1 · latency · cost</text>
</svg>"""


def build_html(m: ExperimentResults) -> str:
    sys, cfg = m.systems, m.config
    best = max(ORDER, key=lambda n: sys[n].accuracy)
    cls, sim = sys["embed-CLASSIFICATION"], sys["embed-SEMANTIC_SIMILARITY"]
    gem, oai = sys[GEMINI_LLM_MODEL_ID], sys[OPENAI_LLM_MODEL_ID]
    cheapest = min(ORDER, key=lambda n: sys[n].cost_usd_per_1k_docs)
    task_delta = cls.accuracy - sim.accuracy

    # Headline comparison for section 1: the stronger LLM against the cheap
    # embedding arm, on both axes the experiment cares about.
    best_llm = max((gem, oai), key=lambda s: s.accuracy)
    acc_gap = cls.accuracy - best_llm.accuracy
    cost_ratio = (
        best_llm.cost_usd_per_1k_docs / cls.cost_usd_per_1k_docs
        if cls.cost_usd_per_1k_docs
        else 0.0
    )

    return f"""<title>Embeddings vs Frontier LLMs</title>
<style>
  :root {{
    --bg: #ffffff; --panel: #fcfcfb; --text: #0b0b0b; --muted: #52514e;
    --grid: #e3e2de; --rule: #ececea;
    --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg: #131312; --panel: #1a1a19; --text: #f5f5f2; --muted: #a5a49c;
      --grid: #383835; --rule: #2a2a28;
      --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
    }}
    :root:not([data-theme="light"]) .light-only {{ display: none; }}
    :root:not([data-theme="light"]) .dark-only {{ display: block; }}
  }}
  :root[data-theme="dark"] {{
    --bg: #131312; --panel: #1a1a19; --text: #f5f5f2; --muted: #a5a49c;
    --grid: #383835; --rule: #2a2a28;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
  }}
  :root[data-theme="dark"] .light-only {{ display: none; }}
  :root[data-theme="dark"] .dark-only {{ display: block; }}
  .dark-only {{ display: none; }}

  body {{ background: var(--bg); color: var(--text); margin: 0;
    font: 16px/1.65 ui-serif, Georgia, serif; }}
  main {{ max-width: 900px; margin: 0 auto; padding: 56px 24px 96px; }}
  h1 {{ font: 700 34px/1.2 system-ui, sans-serif; margin: 0 0 8px; }}
  h2 {{ font: 700 20px/1.3 system-ui, sans-serif; margin: 52px 0 14px;
    padding-top: 18px; border-top: 1px solid var(--rule); }}
  h3 {{ font: 600 15px/1.3 system-ui, sans-serif; margin: 28px 0 8px; }}
  .sub {{ color: var(--muted); font: 15px/1.5 system-ui, sans-serif;
    margin: 0 0 6px; }}
  .q {{ font: 600 19px/1.45 system-ui, sans-serif; color: var(--text);
    margin: 0 0 16px; padding-left: 14px;
    border-left: 3px solid var(--s3); }}
  .byline {{ font: 600 14px system-ui, sans-serif; color: var(--text);
    margin: 0 0 4px; }}
  .defs {{ display: grid; gap: 14px; margin: 20px 0 24px;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
  .def {{ background: var(--panel); border: 1px solid var(--rule);
    border-left: 3px solid var(--s1); border-radius: 8px; padding: 14px 16px; }}
  .def.emb {{ border-left-color: var(--s3); }}
  .def h4 {{ font: 700 14px/1.3 system-ui, sans-serif; margin: 0 0 6px; }}
  .def p {{ font: 14px/1.55 system-ui, sans-serif; margin: 0 0 8px; }}
  .def p:last-child {{ margin-bottom: 0; }}
  .def .out {{ color: var(--muted); font-size: 13px; }}
  .meta {{ color: var(--muted); font: 13px system-ui, sans-serif;
    margin: 0 0 34px; }}
  p {{ margin: 0 0 14px; }}
  code {{ font: 13px ui-monospace, SFMono-Regular, Menlo, monospace;
    background: var(--panel); border: 1px solid var(--rule);
    border-radius: 4px; padding: 1px 5px; }}
  pre {{ background: var(--panel); border: 1px solid var(--rule);
    border-radius: 8px; padding: 14px 16px; overflow-x: auto;
    font: 13px/1.6 ui-monospace, Menlo, monospace; }}
  figure {{ margin: 26px 0; }}
  figure img {{ width: 100%; height: auto; border-radius: 10px;
    border: 1px solid var(--rule); background: var(--panel); }}
  figure svg {{ width: 100%; height: auto; }}
  figcaption {{ color: var(--muted); font: 13px/1.5 system-ui, sans-serif;
    margin-top: 10px; }}
  .tw {{ overflow-x: auto; margin: 22px 0; }}
  table {{ border-collapse: collapse; width: 100%;
    font: 13px/1.5 system-ui, sans-serif; }}
  th {{ text-align: left; color: var(--muted); font-weight: 600;
    border-bottom: 1.5px solid var(--grid); padding: 8px 10px;
    white-space: nowrap; }}
  td {{ border-bottom: 1px solid var(--rule); padding: 8px 10px; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums;
    white-space: nowrap; }}
  .ci {{ color: var(--muted); font-size: 11px; }}
  .cards {{ display: grid; gap: 12px; margin: 26px 0;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }}
  .card {{ background: var(--panel); border: 1px solid var(--rule);
    border-radius: 10px; padding: 14px 16px; }}
  .card .k {{ color: var(--muted); font: 12px system-ui, sans-serif;
    text-transform: uppercase; letter-spacing: .05em; }}
  .card .v {{ font: 700 26px/1.25 system-ui, sans-serif;
    font-variant-numeric: tabular-nums; margin-top: 4px; }}
  .card .n {{ color: var(--muted); font: 12px/1.4 system-ui, sans-serif;
    margin-top: 4px; }}
  ul {{ margin: 0 0 14px; padding-left: 22px; }}
  li {{ margin-bottom: 7px; }}
  .swatch {{ display: inline-block; width: 10px; height: 10px;
    border-radius: 2px; margin-right: 6px; vertical-align: baseline; }}
</style>

<main>
<h1>Embeddings vs Frontier LLMs</h1>
<p class="sub">Can an embedding model that costs
${cls.cost_usd_per_1k_docs:.3f} per 1,000 documents beat a frontier LLM at
sorting news articles into four topics?</p>
<p class="byline">by Xavier Collantes · {cfg.run_date}</p>
<p class="meta">{cfg.n_test} test documents · {cfg.n_train} training
documents · 20 Newsgroups · seed {cfg.seed}</p>

<div class="cards">
  <div class="card"><div class="k">Best accuracy</div>
    <div class="v">{sys[best].accuracy:.1%}</div>
    <div class="n">{SHORT[best].replace(chr(10), ' ')}</div></div>
  <div class="card"><div class="k">Cheapest</div>
    <div class="v">${sys[cheapest].cost_usd_per_1k_docs:.3f}</div>
    <div class="n">per 1,000 docs — {SHORT[cheapest].split(chr(10))[0]}</div></div>
  <div class="card"><div class="k">Task-type delta</div>
    <div class="v">{task_delta:+.1%}</div>
    <div class="n">CLASSIFICATION vs SEMANTIC_SIMILARITY, inside the CI</div></div>
  <div class="card"><div class="k">LLM gap</div>
    <div class="v">{gem.accuracy - oai.accuracy:+.1%}</div>
    <div class="n">Gemini Flash-Lite minus GPT-5.4 nano</div></div>
</div>

<h2>1 · What was tested</h2>
<p class="q">Can 3 AI models classify news articles? Two are LLMs. One is a
super cheap embedding model.</p>

<div class="tw"><table>
<thead><tr>
  <th>Candidate</th><th>Maker</th><th>Released</th><th>Role</th>
  <th style="text-align:right">Price in / out per 1M</th>
  <th style="text-align:right">Accuracy</th>
  <th style="text-align:right">Macro F1</th>
  <th style="text-align:right">$/1k docs</th>
</tr></thead>
<tbody>{candidates_table(m)}</tbody>
</table></div>
<p class="meta">Prices are list prices per 1M tokens; embeddings have no output
charge. The embedding row is the <code>CLASSIFICATION</code> arm; its
<code>SEMANTIC_SIMILARITY</code> control is in section 4.</p>

<h3>LLMs and embedding models do different jobs</h3>
<p>Both read text. Only one of them writes any back.</p>

<div class="defs">
  <div class="def">
    <h4>LLM (large language model)</h4>
    <p>You describe the job in a prompt, "pick one of these four topics", and
    it answers in words. Nobody trained it on your task. It's reading your
    instructions at the moment you ask.</p>
    <p class="out">Output is words. It needs no labelled data at all. You pay
    per token going in and per token coming back, every call, so longer
    articles cost you more.</p>
  </div>
  <div class="def emb">
    <h4>Embedding model</h4>
    <p>It reads text and gives back a list of numbers. {cls.dim:,} of them
    here, placing the article in a space where similar meanings sit close
    together. Ask it a question and you still get numbers. This is what RAG
    and vector search run on: embed once, then compare vectors.</p>
    <p class="out">Output is a vector. To turn vectors into labels you bolt a
    classifier on top, trained here on {cfg.n_train} examples. You pay for
    input tokens only.</p>
  </div>
</div>
<p>The LLM does the whole job on its own and bills you for it. The embedding
model does half the job much cheaper and leaves the other half to you. That
other half is a logistic regression, which costs nothing to fit and nothing to
run.</p>

<p>Quality and cost both matter here, and they pull against each other. The two
LLM arms get the label list and the article in a prompt, then answer in words.
The embedding arm never sees a prompt. It turns each article into a vector with
<code>gemini-embedding-001</code> at <code>task_type=CLASSIFICATION</code>, and
a logistic regression trained on {cfg.n_train} labelled examples reads those
vectors. All three scored the same {cfg.n_test} documents.</p>
<p>The cheap arm won on both counts. It scored
<b>{cls.accuracy:.1%}</b> accuracy at {cls.macro_f1:.3f} macro F1, against
<b>{best_llm.accuracy:.1%}</b> and {best_llm.macro_f1:.3f} for the better of
the two LLMs, a gap of {acc_gap:+.1%}. It also cost
<b>${cls.cost_usd_per_1k_docs:.3f}</b> per 1,000 documents where that LLM cost
<b>${best_llm.cost_usd_per_1k_docs:.3f}</b>, roughly
<b>{cost_ratio:.0f}× more</b>. What the extra money buys is a cold start. The
LLMs needed no labelled data at all, where the embedding arm needed
{cfg.n_train} documents before it could label anything.</p>
<p>A fourth arm runs that same embedding pipeline with one string changed,
<code>task_type=SEMANTIC_SIMILARITY</code>. It's there to check whether the
task type does any work at all.</p>

<figure>{PIPELINE_SVG}
<figcaption>The three pipelines. Only the middle stage differs; the input and
the scoring are identical.</figcaption></figure>

<h3>Setup</h3>
<ul>
  <li><b>Data.</b> 20 Newsgroups with headers, footers and quoted replies
  stripped out. That metadata is what makes this dataset trivially easy.
  Classes:
  {', '.join('<code>' + c + '</code>' for c in cfg.labels)}. Documents
  truncated to {cfg.max_chars:,} characters.</li>
  <li><b>Split.</b> {cfg.n_train} train / {cfg.n_test} test, sampled
  with seed {cfg.seed}. The LLM arms use zero training examples; only the
  embedding arms see the training split.</li>
  <li><b>Prompt.</b> Identical for both LLMs, temperature 0:</li>
</ul>
<pre>{cfg.prompt.replace('{text}', '{{document}}')}</pre>
<ul>
  <li><b>Scoring.</b> Exact-match on the label word; a reply that matches no
  label or several is counted <code>__unknown__</code> and scored wrong, with
  the count reported separately.</li>
  <li><b>Cost.</b> Measured tokens × list price
  (Gemini Flash-Lite ${cfg.pricing_usd_per_1m_tokens[GEMINI_LLM_MODEL_ID].input_per_1m}/
  ${cfg.pricing_usd_per_1m_tokens[GEMINI_LLM_MODEL_ID].output_per_1m},
  GPT-5.4 nano ${cfg.pricing_usd_per_1m_tokens[OPENAI_LLM_MODEL_ID].input_per_1m}/
  ${cfg.pricing_usd_per_1m_tokens[OPENAI_LLM_MODEL_ID].output_per_1m},
  embeddings ${cfg.pricing_usd_per_1m_tokens[EMBED_MODEL_ID].input_per_1m}
  per 1M in/out tokens). Embedding cost counts inference only, since embedding
  the training set is a one-off.</li>
</ul>

<h2>2 · Headline result</h2>
{figure('accuracy_f1', 'Accuracy and macro F1 for all four arms. Error bars are 95% bootstrap confidence intervals on accuracy.')}

<div class="tw"><table>
<thead><tr>
  <th>System</th><th>Kind</th><th>Labelled</th>
  <th style="text-align:right">Accuracy [95% CI]</th>
  <th style="text-align:right">Macro F1</th>
  <th style="text-align:right">Latency/doc</th>
  <th style="text-align:right">$/1k docs</th>
  <th style="text-align:right">Unparsed</th>
</tr></thead>
<tbody>{metrics_table(m)}</tbody>
</table></div>

<h2>3 · Where the errors are</h2>
<p>Aggregate accuracy hides the shape of the mistakes. The per-class F1 chart
shows which topics each system finds hard; the confusion matrices show what it
mistakes them for.</p>
{figure('per_class_f1', 'Per-class F1. A system that is uniformly mediocre looks very different from one that fails a single class.')}
{figure('confusion', 'Confusion matrices, counts, shaded by row-normalised rate. Rows are the true class, columns the prediction; off-diagonal mass is the error structure.')}

<h2>4 · Why the embedding arm works</h2>
<p>The embedding classifier is not doing anything clever. The vectors turn up
already sorted into class-shaped clumps, before any classifier has touched
them, so the logistic regression only has to draw lines between the clumps.
The chart below plots the first two principal components of the test-set
embeddings, with one class lit up per panel.</p>
{figure('embedding_pca', 'PCA of CLASSIFICATION-task embeddings, one class highlighted per panel. Two components of a ' + str(cls.dim) + '-dimensional space, so the real separation is cleaner than this projection makes it look.')}

<h3>Does the task type earn its keyword?</h3>
<p>Same model, same documents, same classifier, one string changed.
<code>CLASSIFICATION</code> scored {cls.accuracy:.1%}
[{cls.accuracy_ci95[0]:.1%}–{cls.accuracy_ci95[1]:.1%}].
<code>SEMANTIC_SIMILARITY</code>, the control, scored {sim.accuracy:.1%}
[{sim.accuracy_ci95[0]:.1%}–{sim.accuracy_ci95[1]:.1%}]. That's
{abs(task_delta):.1%} the wrong way round, and the intervals overlap almost
end to end. {cfg.n_test} documents can't tell these two apart. A control
edging past the thing it was meant to isolate usually means there is nothing
there to find, at least not on four topics this far apart. Pick either one.</p>

<h2>5 · Latency and cost</h2>
{figure('latency', 'Per-document latency. LLM arms are per-call wall clock at concurrency 8; embedding arms are the total embed-plus-predict time amortised over the test set.')}
{figure('cost_accuracy', 'Cost against accuracy. The x-axis is log-scaled, so the horizontal gaps are order-of-magnitude gaps.')}

<h2>6 · What this means</h2>
<ul>
  <li><b>If you have labelled data, embed it.</b> The embedding classifier was
  the cheapest arm by a wide margin and matched the LLMs on accuracy. Its cost
  does not climb with prompt length the way an LLM's does, and it cannot
  return an unparseable answer.</li>
  <li><b>Zero-shot LLMs buy you the cold start.</b> They needed no labelled
  data at all. That is the whole trade, and the embedding arm's
  {cfg.n_train}-document training set is a real prerequisite, not a
  formality.</li>
  <li><b>The task type decided nothing here.</b> The two settings landed
  {abs(task_delta):.1%} apart with confidence intervals sitting on top of each
  other. On four topics this distinct, either one works. Test it again before
  trusting it on a harder label set.</li>
</ul>

<h2>7 · Limitations</h2>
<ul>
  <li>Four well-separated topics. Harder label sets (fine-grained intent,
  overlapping categories, class imbalance) compress the gaps and can flip the
  ordering.</li>
  <li>Zero-shot only. Few-shot prompting or fine-tuning would probably close
  some of the LLM arms' gap, at more tokens per call.</li>
  <li>Single prompt, single run at temperature 0. No prompt-variation study;
  the LLM numbers carry prompt-sensitivity that the CI does not capture.</li>
  <li>Latency was measured over the public internet at concurrency 8 and mixes
  in network and queueing variance.</li>
  <li>Costs are list prices at the time of the run, not billed amounts.</li>
</ul>

<h2>8 · Reproduce</h2>
<pre>python3 -m venv env &amp;&amp; source env/bin/activate
pip install -r requirements.txt
cp .env.example .env      # add GEMINI_API_KEY and OPENAI_API_KEY
python experiment.py      # writes results/metrics.json + predictions.csv
python report.py          # writes results/charts/*.png + results/report.html</pre>
<p class="meta">Raw API responses are cached under <code>results/cache_*.json</code>,
so a re-run is free and byte-identical. <code>python experiment.py --limit 10</code>
smoke-tests the whole pipeline for a few cents.</p>
</main>"""

