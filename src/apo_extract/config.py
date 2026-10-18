"""Pydantic v2 config schema — everything retargetable via YAML, no code edits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class SplitPolicyConfig(BaseModel):
    seed: int = 42
    train_ratio: float = 0.30
    val_ratio: float = 0.40
    test_ratio: float = 0.30

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> "SplitPolicyConfig":
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0 (got {total})")
        return self


class DatasetConfig(BaseModel):
    name: str
    root_dir: str = "data/extract-bench"
    schema_path: str
    documents_glob: str
    gold_dir: str
    gold_match_pattern: str = "{doc_stem}.gold.json"
    split_policy: SplitPolicyConfig = Field(default_factory=SplitPolicyConfig)


class ScoringConfig(BaseModel):
    engine: str = "extract_bench_compliant"
    primary_metric: Literal["overall_micro_f1", "macro_subtree_f1", "mean_metric_score"] = "overall_micro_f1"
    array_alignment_policy: str = "hungarian_plus_llm_cache"
    number_default_tolerance: float = 0.001
    semantic_match_threshold: float = 0.70
    judge_model: str = "gemini/gemini-3.8-flash"
    array_match_threshold: float = 0.5
    enable_llm_judge: bool = False
    cache_path: Optional[str] = "runs/cache/scoring_cache.sqlite"


class LlmConfig(BaseModel):
    provider: str = "gemini"
    model: str
    temperature: float = 0.0
    max_output_tokens: Optional[int] = None
    litellm_model: Optional[str] = None
    pdf_mode: Optional[str] = None  # extractor only: auto | inline | images | text

    @property
    def qualified_model(self) -> str:
        """Model string handed to litellm."""
        if self.litellm_model:
            return self.litellm_model
        if self.provider == "gemini":
            return f"gemini/{self.model}"
        if self.provider in ("vertex", "vertex_ai"):
            return f"vertex_ai/{self.model}"
        if self.provider == "anthropic":
            return f"anthropic/{self.model}"
        if self.provider == "nvidia_nim":
            return f"nvidia_nim/{self.model}"
        return f"{self.provider}/{self.model}"


class LLMsConfig(BaseModel):
    extractor: LlmConfig
    optimizer_critic: Optional[LlmConfig] = None
    optimizer_mutator: Optional[LlmConfig] = None
    scoring_judge: Optional[LlmConfig] = None


class SeedPromptConfig(BaseModel):
    path: str = "configs/prompts/seed_generic_extraction.txt"


class BudgetConfig(BaseModel):
    max_iterations: int = 12
    max_cost_usd: float = 5.0
    max_total_tokens: int = 2_000_000
    max_wall_clock_seconds: int = 1800


class OptimizerConfig(BaseModel):
    strategy: str = "subtree_pareto_beam"
    beam_width: int = 3
    candidates_per_iteration: int = 2
    min_val_improvement: float = 0.005
    stage1_max_train_regression: float = 0.02
    pareto_aggregate_tolerance: float = 0.03
    stall_patience: int = 2
    duplicate_similarity: float = 0.92
    enable_subtree_rollback: bool = True
    rollback_improve_threshold: float = 0.05
    rollback_regress_threshold: float = -0.02
    extraction_mode: Literal["single_call", "subtree_chunked"] = "single_call"
    chunk_threshold_fields: int = 60


class AppConfig(BaseModel):
    run_id: str
    seed: int = 42
    dataset: DatasetConfig
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    llms: LLMsConfig
    seed_prompt: SeedPromptConfig = Field(default_factory=SeedPromptConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    config_path: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        data = yaml.safe_load(Path(path).read_text())
        cfg = cls.model_validate(data)
        cfg.config_path = str(path)
        return cfg

    def config_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()
