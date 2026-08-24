# mrl-exp — Embeddings vs frontier LLMs for text classification

A reproducible experiment: does a cheap embedding classifier beat a frontier
LLM at zero-shot topic classification, and does Gemini's
`task_type=CLASSIFICATION` actually matter?

## The four arms

| Arm | Method | Labelled data |
|---|---|---|
| `gemini-3.5-flash-lite` | zero-shot prompt, temp 0 | 0 |
| `gpt-5.4-nano-2026-03-17` | same prompt, temp 0 | 0 |
| `embed-CLASSIFICATION` | `gemini-embedding-001` (`task_type=CLASSIFICATION`) → logistic regression | 800 |
| `embed-SEMANTIC_SIMILARITY` | identical, but `task_type=SEMANTIC_SIMILARITY` — the control | 800 |

All four score the same 400-document test set: 20 Newsgroups, four classes
(`comp.graphics`, `rec.sport.baseball`, `sci.space`, `sci.electronics`),
headers/footers/quotes stripped, seed 67.

## Run it

```bash
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in GEMINI_API_KEY and OPENAI_API_KEY

python experiment.py --limit 10   # smoke test, a few cents
python experiment.py              # full run, ~800 LLM calls + 2,400 embeddings
python report.py                  # charts + results/report.html
python -m pytest tests/           # unit tests, no API calls
```

`experiment.py` caches every raw API response under `results/cache_*.json`, so
re-running is free and deterministic. Delete the cache files to force fresh calls.

## Layout

| File | What it is |
|---|---|
| `models.py` | pydantic v2 models — the contract for `metrics.json` and the caches |
| `experiment.py` | runs the four arms, writes validated results |
| `report.py` | validates results back in, renders the six charts |
| `template.py` | assembles the charts + numbers into `report.html` |
| `tests/` | label parsing, metric assembly, model validators — no API calls |

`metrics.json` and every `cache_*.json` round-trip through the models, so a
stale or half-written file fails loudly at load rather than producing a wrong
chart. The models also assert what the numbers must mean: accuracy inside its
own confidence interval, `kind` from a closed set, `y_test` matching the
configured row count and label set, and `extra="forbid"` so a renamed field is
an error instead of a silently missing bar.

## Output

- `results/metrics.json` — accuracy, macro F1, per-class P/R/F1, confusion
  matrices, 95% bootstrap CIs, latency, token cost
- `results/predictions.csv` — one row per test document, every arm's prediction
- `results/charts/*.png` — six charts, rendered for light and dark surfaces
- `results/report.html` — the write-up, charts embedded, opens standalone

## Notes on method

- Replies that match no label or several are scored `__unknown__` — wrong, and
  reported separately, rather than silently coerced.
- Cost is measured tokens × list price (constants at the top of
  `experiment.py`; update them when prices move). The embedding arm is charged
  for inference only — embedding the training set is a one-off.
- LLM latency is wall clock at concurrency 8 over the public internet, so it
  carries network variance. Embedding latency is total embed-plus-predict time
  amortised across the test set.
