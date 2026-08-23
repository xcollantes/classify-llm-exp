# mrl-exp — Embeddings vs frontier LLMs for text classification

A reproducible experiment: does a cheap embedding classifier beat a frontier
LLM at zero-shot topic classification, and does Gemini's
`task_type=CLASSIFICATION` actually matter?

## The four arms

| Arm | Method | Labelled data |
|---|---|---|
| `gemini-2.5-flash` | zero-shot prompt, temp 0 | 0 |
| `gpt-4o-mini` | same prompt, temp 0 | 0 |
| `embed-CLASSIFICATION` | `gemini-embedding-001` (`task_type=CLASSIFICATION`) → logistic regression | 800 |
| `embed-SEMANTIC_SIMILARITY` | identical, but `task_type=SEMANTIC_SIMILARITY` — the control | 800 |

All four score the same 400-document test set: 20 Newsgroups, four classes
(`comp.graphics`, `rec.sport.baseball`, `sci.space`, `talk.politics.guns`),
headers/footers/quotes stripped, seed 42.

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
