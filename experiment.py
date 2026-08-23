"""
Text-classification experiment: frontier LLM zero-shot vs Gemini embeddings.

Three systems classify the same 20 Newsgroups test set:
  1. gemini-2.5-flash  zero-shot prompt
  2. gpt-4o-mini       zero-shot prompt (same prompt)
  3. gemini-embedding-001 (task_type=CLASSIFICATION) + LogisticRegression

A fourth arm re-embeds with task_type=SEMANTIC_SIMILARITY as a control, to
isolate the effect of the CLASSIFICATION task type.

Writes results/metrics.json and results/predictions.csv.
Raw API outputs are cached in results/cache_*.json so re-runs are free.
"""

import argparse
import csv
import json
import logging as logger
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

logger.basicConfig(
    level=logger.INFO,
    format="%(asctime)s: %(levelname)s: %(message)s",
)
logging = logger.getLogger(__name__)

RESULTS = Path("results")
SEED = 42
MAX_CHARS = 2000
N_TRAIN = 800
N_TEST = 400
UNKNOWN = "__unknown__"

CATEGORIES = [
    "comp.graphics",
    "rec.sport.baseball",
    "sci.space",
    "talk.politics.guns",
]
LABELS = ["graphics", "baseball", "space", "guns"]
CAT_TO_LABEL = dict(zip(CATEGORIES, LABELS))

GEMINI_LLM = "gemini-2.5-flash"
OPENAI_LLM = "gpt-4o-mini"
EMBED_MODEL = "gemini-embedding-001"

# USD per 1M tokens, list prices recorded 2026-08-22. Update if they move.
PRICING = {
    GEMINI_LLM: {"in": 0.30, "out": 2.50},
    OPENAI_LLM: {"in": 0.15, "out": 0.60},
    EMBED_MODEL: {"in": 0.15, "out": 0.0},
}

PROMPT = (
    "Classify the newsgroup post into exactly one of these topics: "
    + ", ".join(LABELS)
    + ".\nReply with the topic word only, nothing else.\n\nPost:\n{text}"
)


def load_data(n_test: int) -> tuple[list[str], list[str], list[str], list[str]]:
    """Load 20 Newsgroups, stripped of headers/footers/quotes.

    Returns:
        (train_texts, train_labels, test_texts, test_labels)
    """
    rng = random.Random(SEED)
    out = []
    for subset, n in (("train", N_TRAIN), ("test", n_test)):
        bunch = fetch_20newsgroups(
            subset=subset,
            categories=CATEGORIES,
            remove=("headers", "footers", "quotes"),
            random_state=SEED,
        )
        rows = [
            (t.strip()[:MAX_CHARS], CAT_TO_LABEL[bunch.target_names[y]])
            for t, y in zip(bunch.data, bunch.target)
            if len(t.strip()) > 50
        ]
        rng.shuffle(rows)
        rows = rows[:n]
        out.append([r[0] for r in rows])
        out.append([r[1] for r in rows])
    return tuple(out)


def parse_label(reply: str) -> str:
    """Map a free-text model reply onto a label, or UNKNOWN."""
    low = reply.strip().lower()
    for label in LABELS:
        if low == label:
            return label
    hits = [label for label in LABELS if label in low]
    return hits[0] if len(hits) == 1 else UNKNOWN


def _retry(fn, tries: int = 5):
    """Call fn with exponential backoff on any exception."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:  # Reason: 429/503 are the common ones.
            if attempt == tries - 1:
                raise
            wait = 2 ** attempt
            logging.warning("retry %d after %ss: %s", attempt + 1, wait, exc)
            time.sleep(wait)


def call_gemini(client, text: str) -> dict:
    """One zero-shot classification call to Gemini."""
    from google.genai import types

    started = time.perf_counter()
    resp = _retry(lambda: client.models.generate_content(
        model=GEMINI_LLM,
        contents=PROMPT.format(text=text),
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=800,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    ))
    usage = resp.usage_metadata
    return {
        "reply": (resp.text or "").strip(),
        "latency_s": time.perf_counter() - started,
        "in_tokens": usage.prompt_token_count or 0,
        "out_tokens": usage.candidates_token_count or 0,
    }


def call_openai(client, text: str) -> dict:
    """One zero-shot classification call to OpenAI."""
    started = time.perf_counter()
    resp = _retry(lambda: client.chat.completions.create(
        model=OPENAI_LLM,
        temperature=0,
        max_tokens=10,
        messages=[{"role": "user", "content": PROMPT.format(text=text)}],
    ))
    return {
        "reply": (resp.choices[0].message.content or "").strip(),
        "latency_s": time.perf_counter() - started,
        "in_tokens": resp.usage.prompt_tokens,
        "out_tokens": resp.usage.completion_tokens,
    }


def run_llm(name: str, call, client, texts: list[str]) -> list[dict]:
    """Classify every text, caching raw results to disk."""
    cache = RESULTS / f"cache_{name}.json"
    if cache.exists():
        rows = json.loads(cache.read_text())
        if len(rows) == len(texts):
            logging.info("%s: %d cached results reused", name, len(rows))
            return rows
    logging.info("%s: calling API for %d docs", name, len(texts))
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda t: call(client, t), texts))
    cache.write_text(json.dumps(rows))
    return rows


def embed(client, texts: list[str], task_type: str) -> np.ndarray:
    """Embed texts with gemini-embedding-001, cached to disk."""
    from google.genai import types

    cache = RESULTS / f"cache_embed_{task_type.lower()}_{len(texts)}.json"
    if cache.exists():
        logging.info("embed %s: %d cached vectors reused", task_type, len(texts))
        return np.array(json.loads(cache.read_text()))
    logging.info("embed %s: %d docs", task_type, len(texts))
    vectors = []
    for i in range(0, len(texts), 50):
        batch = texts[i:i + 50]
        resp = _retry(lambda: client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        ))
        vectors.extend(e.values for e in resp.embeddings)
    arr = np.array(vectors)
    arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
    cache.write_text(json.dumps(arr.tolist()))
    return arr


def bootstrap_ci(y_true: list[str], y_pred: list[str], n: int = 2000) -> list[float]:
    """95% bootstrap confidence interval on accuracy."""
    rng = np.random.default_rng(SEED)
    correct = np.array([t == p for t, p in zip(y_true, y_pred)], dtype=float)
    means = [correct[rng.integers(0, len(correct), len(correct))].mean()
             for _ in range(n)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def score(name: str, y_true: list[str], y_pred: list[str],
          latencies: list[float], cost: float, extra: dict) -> dict:
    """Build the metrics record for one system."""
    report = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "name": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS,
                             average="macro", zero_division=0),
        "accuracy_ci95": bootstrap_ci(y_true, y_pred),
        "per_class_f1": {c: report[c]["f1-score"] for c in LABELS},
        "per_class_precision": {c: report[c]["precision"] for c in LABELS},
        "per_class_recall": {c: report[c]["recall"] for c in LABELS},
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "unparsed": sum(p == UNKNOWN for p in y_pred),
        "latency_mean_s": float(np.mean(latencies)) if latencies else 0.0,
        "latency_p95_s": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "latencies": latencies,
        "cost_usd_per_1k_docs": cost / len(y_true) * 1000,
        **extra,
    }


def token_cost(rows: list[dict], model: str) -> float:
    """Total USD for a batch of LLM calls."""
    price = PRICING[model]
    return sum(r["in_tokens"] * price["in"] + r["out_tokens"] * price["out"]
               for r in rows) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=N_TEST,
                        help="number of test docs (smoke test with e.g. 10)")
    args = parser.parse_args()

    load_dotenv()
    from google import genai
    from openai import OpenAI

    gclient = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    oclient = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    x_train, y_train, x_test, y_test = load_data(args.limit)
    logging.info("train=%d test=%d", len(x_train), len(x_test))

    systems = {}

    # --- Arm 1 & 2: zero-shot LLMs -------------------------------------
    for name, model, call, client in (
        ("gemini-2.5-flash", GEMINI_LLM, call_gemini, gclient),
        ("gpt-4o-mini", OPENAI_LLM, call_openai, oclient),
    ):
        rows = run_llm(name, call, client, x_test)
        preds = [parse_label(r["reply"]) for r in rows]
        systems[name] = score(
            name, y_test, preds, [r["latency_s"] for r in rows],
            token_cost(rows, model),
            {"kind": "llm-zero-shot", "labeled_examples": 0},
        )
        systems[name]["preds"] = preds

    # --- Arm 3 & 4: embeddings + logistic regression --------------------
    for name, task in (
        ("embed-CLASSIFICATION", "CLASSIFICATION"),
        ("embed-SEMANTIC_SIMILARITY", "SEMANTIC_SIMILARITY"),
    ):
        started = time.perf_counter()
        train_vec = embed(gclient, x_train, task)
        test_vec = embed(gclient, x_test, task)
        clf = LogisticRegression(max_iter=1000, random_state=SEED)
        clf.fit(train_vec, y_train)
        preds = list(clf.predict(test_vec))
        elapsed = time.perf_counter() - started
        # Embedding cost: ~1 token per 4 chars, inference-side docs only.
        infer_tokens = sum(len(t) for t in x_test) / 4
        cost = infer_tokens * PRICING[EMBED_MODEL]["in"] / 1e6
        systems[name] = score(
            name, y_test, preds, [elapsed / len(x_test)] * len(x_test), cost,
            {"kind": "embedding+logreg", "labeled_examples": len(x_train),
             "dim": int(train_vec.shape[1])},
        )
        systems[name]["preds"] = preds
        if task == "CLASSIFICATION":
            np.save(RESULTS / "test_embeddings.npy", test_vec)

    # --- Persist --------------------------------------------------------
    with open(RESULTS / "predictions.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        names = list(systems)
        writer.writerow(["doc_id", "true"] + names + ["text_preview"])
        for i, truth in enumerate(y_test):
            writer.writerow([i, truth] + [systems[n]["preds"][i] for n in names]
                            + [x_test[i][:120].replace("\n", " ")])

    metrics = {
        "config": {
            "dataset": "20newsgroups (headers/footers/quotes removed)",
            "labels": LABELS,
            "n_train": len(x_train),
            "n_test": len(x_test),
            "seed": SEED,
            "max_chars": MAX_CHARS,
            "prompt": PROMPT,
            "pricing_usd_per_1m_tokens": PRICING,
        },
        "systems": {n: {k: v for k, v in s.items() if k != "preds"}
                    for n, s in systems.items()},
        "y_test": y_test,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    for name, s in systems.items():
        logging.info("%-28s acc=%.3f  macroF1=%.3f  $%.3f/1k docs",
                     name, s["accuracy"], s["macro_f1"],
                     s["cost_usd_per_1k_docs"])


if __name__ == "__main__":
    main()
