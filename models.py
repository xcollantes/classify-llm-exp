"""Pydantic models for the experiment's on-disk data.

These are the contract between experiment.py (writes) and report.py (reads):
results/metrics.json and results/cache_*.json both validate against them, so a
stale or half-written file fails loudly at load instead of producing a wrong
chart.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelPrice(BaseModel):
    """List price for one model, USD per 1M tokens."""

    model_config = ConfigDict(extra="forbid")

    input_per_1m: float = Field(ge=0)
    output_per_1m: float = Field(ge=0)


class LlmCall(BaseModel):
    """One LLM classification call: the reply plus what it cost."""

    model_config = ConfigDict(extra="forbid")

    reply: str
    latency_s: float = Field(ge=0)
    in_tokens: int = Field(ge=0)
    out_tokens: int = Field(ge=0)


class SystemMetrics(BaseModel):
    """Scored results for one arm of the experiment."""

    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["llm-zero-shot", "embedding+logreg"]
    labeled_examples: int = Field(ge=0)
    accuracy: float = Field(ge=0, le=1)
    macro_f1: float = Field(ge=0, le=1)
    accuracy_ci95: tuple[float, float]
    per_class_f1: dict[str, float]
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    confusion: list[list[int]]
    unparsed: int = Field(ge=0)
    latency_mean_s: float = Field(ge=0)
    latency_p95_s: float = Field(ge=0)
    latencies: list[float]
    cost_usd_per_1k_docs: float = Field(ge=0)
    dim: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def check_ci_brackets_accuracy(self) -> "SystemMetrics":
        """A CI that does not contain its point estimate means a scoring bug."""
        low, high = self.accuracy_ci95
        if not low <= self.accuracy <= high:
            raise ValueError(
                f"{self.name}: accuracy {self.accuracy} outside CI [{low}, {high}]")
        return self


class ExperimentConfig(BaseModel):
    """Everything needed to reproduce the run."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    labels: list[str] = Field(min_length=2)
    n_train: int = Field(ge=0)
    n_test: int = Field(gt=0)
    seed: int
    max_chars: int = Field(gt=0)
    prompt: str
    pricing_usd_per_1m_tokens: dict[str, ModelPrice]


class ExperimentResults(BaseModel):
    """The whole of results/metrics.json."""

    model_config = ConfigDict(extra="forbid")

    config: ExperimentConfig
    systems: dict[str, SystemMetrics]
    y_test: list[str]

    @model_validator(mode="after")
    def check_row_counts(self) -> "ExperimentResults":
        """Every arm must have scored the same test set."""
        if len(self.y_test) != self.config.n_test:
            raise ValueError(
                f"y_test has {len(self.y_test)} rows, config says "
                f"{self.config.n_test}")
        unknown = set(self.y_test) - set(self.config.labels)
        if unknown:
            raise ValueError(f"y_test has labels outside the label set: {unknown}")
        return self
