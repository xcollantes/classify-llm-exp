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
import textwrap
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from dotenv import load_dotenv
from pydantic import TypeAdapter
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils import Bunch

from models import (
    ExperimentConfig,
    ExperimentResults,
    LlmCall,
    ModelPrice,
    SystemMetrics,
)

if TYPE_CHECKING:  # Reason: SDKs are imported lazily inside main().
    from google.genai import Client as GeminiClient
    from openai import OpenAI

T = TypeVar("T")
CALLS: TypeAdapter[list[LlmCall]] = TypeAdapter(list[LlmCall])

logger.basicConfig(
    level=logger.INFO,
    format="%(asctime)s: %(levelname)s: %(message)s",
)
logging = logger.getLogger(__name__)

RESULTS: Path = Path("results")
SEED: int = 67
MAX_CHARS: int = 2000
N_TRAIN: int = 800
N_TEST: int = 400
UNKNOWN: str = "__unknown__"

CATEGORIES: list[str] = [
    "comp.graphics",
    "rec.sport.baseball",
    "sci.space",
    "sci.electronics",
]
LABELS: list[str] = ["graphics", "baseball", "space", "electronics"]
CAT_TO_LABEL: dict[str, str] = dict(zip(CATEGORIES, LABELS))

GEMINI_LLM: str = "gemini-2.5-flash"
OPENAI_LLM: str = "gpt-4o-mini"
EMBED_MODEL: str = "gemini-embedding-001"

# List prices as of 2026-08-22
PRICING: dict[str, ModelPrice] = {
    GEMINI_LLM: ModelPrice(input_per_1m=0.30, output_per_1m=2.50),
    OPENAI_LLM: ModelPrice(input_per_1m=0.15, output_per_1m=0.60),
    EMBED_MODEL: ModelPrice(input_per_1m=0.15, output_per_1m=0.0),
}

PROMPT: str = textwrap.dedent(
    f"""
    Classify the news article into exactly one of these topics:
    {", ".join(LABELS)}
    Reply with the topic word only, nothing else.
    Article:
    {{text}}
    """
)


def load_data(n_test: int) -> tuple[list[str], list[str], list[str], list[str]]:
    """Load news articles omit headers/footers/quotes.

    Args:
        n_test: Number of test documents to keep from skikit

    Returns:
        (train_texts, train_labels, test_texts, test_labels)
    """
    rng: random.Random = random.Random(SEED)

    splits: list[list[tuple[str, str]]] = []

    for subset, n in (("train", N_TRAIN), ("test", n_test)):
        bunch: Bunch = fetch_20newsgroups(
            # Divided into train and test as per Kaggle
            subset=subset,
            categories=CATEGORIES,
            # EXP_NOTE: try to remove noise
            remove=("headers", "footers", "quotes"),
            random_state=SEED,
        )

        rows: list[tuple[str, str]] = [
            (t.strip()[:MAX_CHARS], CAT_TO_LABEL[bunch.target_names[y]])
            for t, y in zip(bunch.data, bunch.target)
            if len(t.strip()) > 50
        ]

        rng.shuffle(rows)
        splits.append(rows[:n])

    train, test = splits

    return (
        [t for t, _ in train],
        [y for _, y in train],
        [t for t, _ in test],
        [y for _, y in test],
    )


def parse_label(reply: str) -> str:
    """Map a free-text model reply onto a label, or UNKNOWN.

    Args:
        reply: Raw text the model returned.

    Returns:
        One of LABELS, or UNKNOWN when the reply matches none or several.
    """
    low = reply.strip().lower()
    for label in LABELS:
        if low == label:
            return label
    hits = [label for label in LABELS if label in low]
    return hits[0] if len(hits) == 1 else UNKNOWN


def _retry(fn: Callable[[], T], tries: int = 5) -> T:
    """Call fn with exponential backoff on any exception.

    Args:
        fn: Zero-argument callable to invoke.
        tries: Total attempts before giving up.

    Returns:
        Whatever fn returns on its first successful call.

    Raises:
        Exception: The last failure, once tries is exhausted.
    """
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:  # Reason: 429/503 are the common ones.
            if attempt == tries - 1:
                raise
            wait = 2**attempt
            logging.warning("retry %d after %ss: %s", attempt + 1, wait, exc)
            time.sleep(wait)
    raise RuntimeError("unreachable: loop always returns or raises")


def call_gemini(client: "GeminiClient", text: str) -> LlmCall:
    """One zero-shot classification call to Gemini."""
    from google.genai import types

    started = time.perf_counter()
    resp = _retry(
        lambda: client.models.generate_content(
            model=GEMINI_LLM,
            contents=PROMPT.format(text=text),
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=800,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
    )
    usage = resp.usage_metadata
    return LlmCall(
        reply=(resp.text or "").strip(),
        latency_s=time.perf_counter() - started,
        in_tokens=usage.prompt_token_count or 0,
        out_tokens=usage.candidates_token_count or 0,
    )


def call_openai(client: "OpenAI", text: str) -> LlmCall:
    """One zero-shot classification call to OpenAI."""
    started = time.perf_counter()
    resp = _retry(
        lambda: client.chat.completions.create(
            model=OPENAI_LLM,
            temperature=0,
            max_tokens=10,
            messages=[{"role": "user", "content": PROMPT.format(text=text)}],
        )
    )
    return LlmCall(
        reply=(resp.choices[0].message.content or "").strip(),
        latency_s=time.perf_counter() - started,
        in_tokens=resp.usage.prompt_tokens,
        out_tokens=resp.usage.completion_tokens,
    )


def run_llm(
    name: str,
    call: Callable[[Any, str], LlmCall],
    client: Any,
    texts: list[str],
) -> list[LlmCall]:
    """Classify every text, caching raw results to disk.

    Args:
        name: Cache key and log label for this arm.
        call: Per-document classification function.
        client: SDK client handed to call.
        texts: Documents to classify.

    Returns:
        One LlmCall per input document, in order.
    """
    cache = RESULTS / f"cache_{name}.json"
    if cache.exists():
        rows = CALLS.validate_json(cache.read_bytes())
        if len(rows) == len(texts):
            logging.info("%s: %d cached results reused", name, len(rows))
            return rows
    logging.info("%s: calling API for %d docs", name, len(texts))
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda t: call(client, t), texts))
    cache.write_bytes(CALLS.dump_json(rows))
    return rows


def embed(client: "GeminiClient", texts: list[str], task_type: str) -> np.ndarray:
    """Embed texts with gemini-embedding-001, cached to disk.

    Args:
        client: Gemini SDK client.
        texts: Documents to embed.
        task_type: Embedding task type, e.g. "CLASSIFICATION".

    Returns:
        L2-normalised embedding matrix, one row per document.
    """
    from google.genai import types

    cache = RESULTS / f"cache_embed_{task_type.lower()}_{len(texts)}.json"
    if cache.exists():
        logging.info("embed %s: %d cached vectors reused", task_type, len(texts))
        return np.array(json.loads(cache.read_text()))
    logging.info("embed %s: %d docs", task_type, len(texts))
    vectors: list[list[float]] = []
    for i in range(0, len(texts), 50):
        batch = texts[i : i + 50]
        resp = _retry(
            lambda: client.models.embed_content(
                model=EMBED_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
        )
        vectors.extend(e.values for e in resp.embeddings)
    arr = np.array(vectors)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def bootstrap_ci(
    y_true: list[str], y_pred: list[str], n: int = 2000
) -> tuple[float, float]:
    """95% bootstrap confidence interval on accuracy.

    Args:
        y_true: Gold labels.
        y_pred: Predicted labels, same order.
        n: Number of bootstrap resamples.

    Returns:
        (lower, upper) percentile bounds.
    """
    rng = np.random.default_rng(SEED)
    correct = np.array([t == p for t, p in zip(y_true, y_pred)], dtype=float)
    means = [
        correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(n)
    ]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def score(
    name: str,
    y_true: list[str],
    y_pred: list[str],
    latencies: list[float],
    cost: float,
    kind: Literal["llm-zero-shot", "embedding+logreg"],
    labeled_examples: int,
    dim: int | None = None,
) -> SystemMetrics:
    """Build the validated metrics record for one system.

    Args:
        name: Arm name.
        y_true: Gold labels.
        y_pred: Predicted labels, same order.
        latencies: Per-document seconds.
        cost: Total USD spent classifying this test set.
        kind: Which family of method this arm belongs to.
        labeled_examples: Training documents the arm consumed.
        dim: Embedding dimensionality, for embedding arms only.

    Returns:
        A SystemMetrics instance; construction validates the numbers.
    """
    report = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    return SystemMetrics(
        name=name,
        kind=kind,
        labeled_examples=labeled_examples,
        accuracy=accuracy_score(y_true, y_pred),
        macro_f1=f1_score(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        ),
        accuracy_ci95=bootstrap_ci(y_true, y_pred),
        per_class_f1={c: report[c]["f1-score"] for c in LABELS},
        per_class_precision={c: report[c]["precision"] for c in LABELS},
        per_class_recall={c: report[c]["recall"] for c in LABELS},
        confusion=confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        unparsed=sum(p == UNKNOWN for p in y_pred),
        latency_mean_s=float(np.mean(latencies)) if latencies else 0.0,
        latency_p95_s=float(np.percentile(latencies, 95)) if latencies else 0.0,
        latencies=latencies,
        cost_usd_per_1k_docs=cost / len(y_true) * 1000,
        dim=dim,
    )


def token_cost(rows: list[LlmCall], model: str) -> float:
    """Total USD for a batch of LLM calls."""
    price = PRICING[model]
    return (
        sum(
            r.in_tokens * price.input_per_1m + r.out_tokens * price.output_per_1m
            for r in rows
        )
        / 1e6
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=N_TEST,
        help="number of test docs (smoke test with e.g. 10)",
    )
    args = parser.parse_args()

    load_dotenv()
    from google import genai
    from openai import OpenAI

    gclient = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    oclient = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    x_train, y_train, x_test, y_test = load_data(args.limit)
    logging.info("train=%d test=%d", len(x_train), len(x_test))

    systems: dict[str, SystemMetrics] = {}
    predictions: dict[str, list[str]] = {}

    # --- Arm 1 & 2: zero-shot LLMs -------------------------------------
    for name, model, call, client in (
        ("gemini-2.5-flash", GEMINI_LLM, call_gemini, gclient),
        ("gpt-4o-mini", OPENAI_LLM, call_openai, oclient),
    ):
        rows = run_llm(name, call, client, x_test)
        preds = [parse_label(r.reply) for r in rows]
        systems[name] = score(
            name,
            y_test,
            preds,
            [r.latency_s for r in rows],
            token_cost(rows, model),
            kind="llm-zero-shot",
            labeled_examples=0,
        )
        predictions[name] = preds

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
        # Reason: ~1 token per 4 chars; inference-side docs only, since
        # embedding the training set is a one-off cost.
        infer_tokens = sum(len(t) for t in x_test) / 4
        cost = infer_tokens * PRICING[EMBED_MODEL].input_per_1m / 1e6
        systems[name] = score(
            name,
            y_test,
            preds,
            [elapsed / len(x_test)] * len(x_test),
            cost,
            kind="embedding+logreg",
            labeled_examples=len(x_train),
            dim=int(train_vec.shape[1]),
        )
        predictions[name] = preds
        if task == "CLASSIFICATION":
            np.save(RESULTS / "test_embeddings.npy", test_vec)

    # --- Persist --------------------------------------------------------
    with open(RESULTS / "predictions.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        names = list(systems)
        writer.writerow(["doc_id", "true"] + names + ["text_preview"])
        for i, truth in enumerate(y_test):
            writer.writerow(
                [i, truth]
                + [predictions[n][i] for n in names]
                + [x_test[i][:120].replace("\n", " ")]
            )

    results = ExperimentResults(
        config=ExperimentConfig(
            dataset="20newsgroups (headers/footers/quotes removed)",
            labels=LABELS,
            n_train=len(x_train),
            n_test=len(x_test),
            seed=SEED,
            max_chars=MAX_CHARS,
            prompt=PROMPT,
            pricing_usd_per_1m_tokens=PRICING,
        ),
        systems=systems,
        y_test=y_test,
    )
    (RESULTS / "metrics.json").write_text(results.model_dump_json(indent=2))

    for name, s in systems.items():
        logging.info(
            "%-28s acc=%.3f  macroF1=%.3f  $%.3f/1k docs",
            name,
            s.accuracy,
            s.macro_f1,
            s.cost_usd_per_1k_docs,
        )


if __name__ == "__main__":
    main()
