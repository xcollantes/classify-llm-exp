"""Tests for the pydantic contract between experiment.py and report.py."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import ExperimentConfig, ExperimentResults, ModelInfo, SystemMetrics


def make_metrics(**overrides: object) -> SystemMetrics:
    """Build a valid SystemMetrics, with fields overridden for the test."""
    base = dict(
        name="arm",
        kind="llm-zero-shot",
        labeled_examples=0,
        accuracy=0.9,
        macro_f1=0.9,
        accuracy_ci95=(0.85, 0.95),
        per_class_f1={"a": 0.9},
        per_class_precision={"a": 0.9},
        per_class_recall={"a": 0.9},
        confusion=[[9, 1], [1, 9]],
        unparsed=0,
        latency_mean_s=0.5,
        latency_p95_s=0.9,
        latencies=[0.5],
        cost_usd_per_1k_docs=0.2,
    )
    return SystemMetrics(**{**base, **overrides})


def make_config(**overrides: object) -> ExperimentConfig:
    """Build a valid ExperimentConfig, with fields overridden for the test."""
    base = dict(
        dataset="20newsgroups",
        run_date="2026-01-01",
        labels=["a", "b"],
        n_train=800,
        n_test=2,
        seed=42,
        max_chars=2000,
        prompt="classify: {text}",
        pricing_usd_per_1m_tokens={
            "m": ModelInfo(
                name="m",
                model="m",
                maker="ACME",
                released="2026-01-01",
                input_per_1m=0.1,
                output_per_1m=0.2,
            )
        },
    )
    return ExperimentConfig(**{**base, **overrides})


def test_valid_metrics_round_trip() -> None:
    """A valid record survives serialisation unchanged."""
    m = make_metrics()
    assert SystemMetrics.model_validate_json(m.model_dump_json()) == m


def test_accuracy_outside_ci_is_rejected() -> None:
    """A CI that misses its point estimate means a scoring bug, not a warning."""
    with pytest.raises(ValidationError, match="outside CI"):
        make_metrics(accuracy=0.5, accuracy_ci95=(0.85, 0.95))


def test_accuracy_above_one_is_rejected() -> None:
    """Accuracy is a proportion."""
    with pytest.raises(ValidationError):
        make_metrics(accuracy=1.4, accuracy_ci95=(0.85, 1.5))


def test_unknown_kind_is_rejected() -> None:
    """kind is a closed set — a typo must not reach the charts."""
    with pytest.raises(ValidationError):
        make_metrics(kind="magic")


def test_extra_field_is_rejected() -> None:
    """extra='forbid' catches renamed fields at load instead of at chart time."""
    with pytest.raises(ValidationError):
        make_metrics(preds=["a", "b"])


def test_results_reject_wrong_test_set_size() -> None:
    """y_test must match the row count the config claims."""
    with pytest.raises(ValidationError, match="config says"):
        ExperimentResults(
            config=make_config(n_test=3),
            systems={"arm": make_metrics()},
            y_test=["a", "b"],
        )


def test_results_reject_label_outside_label_set() -> None:
    """A stray label means the split and the config disagree."""
    with pytest.raises(ValidationError, match="outside the label set"):
        ExperimentResults(
            config=make_config(),
            systems={"arm": make_metrics()},
            y_test=["a", "zzz"],
        )


def test_valid_results_round_trip() -> None:
    """The whole metrics.json shape survives a dump/load cycle."""
    r = ExperimentResults(
        config=make_config(),
        systems={"arm": make_metrics()},
        y_test=["a", "b"],
    )
    assert ExperimentResults.model_validate_json(r.model_dump_json()) == r
