"""Unit tests for label parsing and metric assembly."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiment import LABELS, UNKNOWN, bootstrap_ci, parse_label, score


@pytest.mark.parametrize("reply,expected", [
    ("space", "space"),
    ("  Space\n", "space"),
    ("The topic is guns.", "guns"),
    ("baseball", "baseball"),
    ("I think this is about space or guns", UNKNOWN),
    ("", UNKNOWN),
    ("hockey", UNKNOWN),
])
def test_parse_label(reply: str, expected: str) -> None:
    """Replies map to a single label, or UNKNOWN when ambiguous."""
    assert parse_label(reply) == expected


def test_score_perfect_predictions() -> None:
    """A perfect run scores 1.0 with no unparsed replies."""
    truth = LABELS * 5
    result = score("perfect", truth, truth, [0.1] * 20, 0.02,
                   kind="llm-zero-shot", labeled_examples=0)
    assert result.accuracy == 1.0
    assert result.macro_f1 == 1.0
    assert result.unparsed == 0
    assert result.cost_usd_per_1k_docs == pytest.approx(1.0)


def test_score_counts_unknown_as_wrong() -> None:
    """UNKNOWN predictions are wrong and reported separately."""
    truth = ["space", "guns", "space", "guns"]
    preds = ["space", UNKNOWN, "space", "guns"]
    result = score("partial", truth, preds, [0.1] * 4, 0.0,
                   kind="llm-zero-shot", labeled_examples=0)
    assert result.accuracy == 0.75
    assert result.unparsed == 1


def test_bootstrap_ci_brackets_accuracy() -> None:
    """The CI contains the point estimate and stays in [0, 1]."""
    truth = ["space"] * 80 + ["guns"] * 20
    preds = ["space"] * 100
    lo, hi = bootstrap_ci(truth, preds, n=500)
    assert 0.0 <= lo <= 0.8 <= hi <= 1.0
