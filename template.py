"""HTML assembly for the experiment report.

Charts are rendered by report.py; this module turns the validated results plus
those PNGs into the single self-contained results/report.html.
"""

import base64
from pathlib import Path

from models import ExperimentResults, SystemMetrics

CHARTS: Path = Path("results") / "charts"

ORDER: list[str] = [
    "gemini-2.5-flash",
    "gpt-4o-mini",
    "embed-CLASSIFICATION",
    "embed-SEMANTIC_SIMILARITY",
]
SHORT: dict[str, str] = {
    "gemini-2.5-flash": "Gemini 2.5 Flash\n(zero-shot)",
    "gpt-4o-mini": "GPT-4o mini\n(zero-shot)",
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
  <text class="lb" x="419" y="40" text-anchor="middle">Gemini 2.5 Flash</text>
  <text class="sb" x="419" y="58" text-anchor="middle">zero-shot prompt, temp 0</text>

  <rect class="bx" x="304" y="96" width="230" height="56" style="stroke:var(--s2)"/>
  <text class="lb" x="419" y="118" text-anchor="middle">GPT-4o mini</text>
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
    gem, oai = sys["gemini-2.5-flash"], sys["gpt-4o-mini"]
    cheapest = min(ORDER, key=lambda n: sys[n].cost_usd_per_1k_docs)
    task_delta = cls.accuracy - sim.accuracy

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
<p class="sub">Does a $0.0002 embedding classifier beat a frontier LLM at
picking one of four topics?</p>
<p class="meta">{cfg.n_test} test documents · {cfg.n_train} training
documents · 20 Newsgroups · seed {cfg.seed}</p>

<div class="cards">
  <div class="card"><div class="k">Best accuracy</div>
    <div class="v">{sys[best].accuracy:.1%}</div>
    <div class="n">{SHORT[best].replace(chr(10), ' ')}</div></div>
  <div class="card"><div class="k">Cheapest</div>
    <div class="v">${sys[cheapest].cost_usd_per_1k_docs:.3f}</div>
    <div class="n">per 1,000 docs — {SHORT[cheapest].split(chr(10))[0]}</div></div>
  <div class="card"><div class="k">Task-type gain</div>
    <div class="v">{task_delta:+.1%}</div>
    <div class="n">CLASSIFICATION over SEMANTIC_SIMILARITY</div></div>
  <div class="card"><div class="k">LLM gap</div>
    <div class="v">{gem.accuracy - oai.accuracy:+.1%}</div>
    <div class="n">Gemini Flash minus GPT-4o mini</div></div>
</div>

<h2>1 · What was tested</h2>
<p>Three ways to put a topic label on a piece of text, run over the identical
{cfg.n_test}-document test set. Two are zero-shot frontier LLMs given the
label list and nothing else. The third never sees a prompt: it turns each
document into a vector with <code>gemini-embedding-001</code> at
<code>task_type=CLASSIFICATION</code> and hands those vectors to a logistic
regression trained on {cfg.n_train} labelled examples.</p>
<p>A fourth arm re-runs the embedding pipeline unchanged except for
<code>task_type=SEMANTIC_SIMILARITY</code>. It exists to answer a narrower
question: does the task type actually do anything, or is it decoration?</p>

<figure>{PIPELINE_SVG}
<figcaption>The three pipelines. Only the middle stage differs; the input and
the scoring are identical.</figcaption></figure>

<h3>Setup</h3>
<ul>
  <li><b>Data.</b> 20 Newsgroups with headers, footers and quoted replies
  stripped — the metadata that makes this dataset trivially easy. Classes:
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
  (Gemini Flash ${cfg.pricing_usd_per_1m_tokens['gemini-2.5-flash'].input_per_1m}/
  ${cfg.pricing_usd_per_1m_tokens['gemini-2.5-flash'].output_per_1m},
  GPT-4o mini ${cfg.pricing_usd_per_1m_tokens['gpt-4o-mini'].input_per_1m}/
  ${cfg.pricing_usd_per_1m_tokens['gpt-4o-mini'].output_per_1m},
  embeddings ${cfg.pricing_usd_per_1m_tokens['gemini-embedding-001'].input_per_1m}
  per 1M in/out tokens). Embedding cost counts inference only — training-set
  embedding is a one-off.</li>
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
<p>The embedding classifier is not doing anything clever. With
<code>task_type=CLASSIFICATION</code> the vectors already arrange themselves
into class-shaped regions before any classifier is fitted — the logistic
regression only has to draw the boundaries. Below is the first two principal
components of the test-set embeddings, faceted so each class is highlighted
against the rest.</p>
{figure('embedding_pca', 'PCA of CLASSIFICATION-task embeddings, one class highlighted per panel. Two components of a ' + str(cls.dim) + '-dimensional space — real separation is cleaner than this projection makes it look.')}

<h3>The task type is not decoration</h3>
<p>Same model, same documents, same classifier, one string changed:
<code>CLASSIFICATION</code> scored {cls.accuracy:.1%} against
{sim.accuracy:.1%} for <code>SEMANTIC_SIMILARITY</code>, a
{task_delta:+.1%} difference.</p>

<h2>5 · Latency and cost</h2>
{figure('latency', 'Per-document latency. LLM arms are per-call wall clock at concurrency 8; embedding arms are the total embed-plus-predict time amortised over the test set.')}
{figure('cost_accuracy', 'Cost against accuracy. The x-axis is log-scaled — the horizontal gaps are order-of-magnitude gaps.')}

<h2>6 · What this means</h2>
<ul>
  <li><b>If you have labelled data, embed it.</b> The embedding classifier is
  the cheapest arm by a wide margin and competitive on accuracy. Its cost does
  not grow with prompt length the way an LLM's does, and it has no parsing
  failure mode.</li>
  <li><b>Zero-shot LLMs buy you the cold start.</b> They needed no labelled
  data at all. That is the whole trade: the embedding arm's
  {cfg.n_train}-document training set is a real prerequisite.</li>
  <li><b>Set the task type.</b> It is one keyword argument and it moved
  accuracy by {abs(task_delta):.1%} here.</li>
</ul>

<h2>7 · Limitations</h2>
<ul>
  <li>Four well-separated topics. Harder label sets — fine-grained intent,
  overlapping categories, class imbalance — compress the gaps and can flip the
  ordering.</li>
  <li>Zero-shot only. Few-shot prompting or fine-tuning would likely close some
  of the LLM arms' gap, at more tokens per call.</li>
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

