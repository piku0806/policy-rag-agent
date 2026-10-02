"""MLflow tracing setup. Local SQLite by default; point MLFLOW_TRACKING_URI at Databricks in production."""
from __future__ import annotations

import os
import warnings

import mlflow

from policyrag import llm


def setup(experiment: str = "policy-rag-agent") -> None:
    if not os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    if not os.getenv("MLFLOW_EXPERIMENT_ID"):
        mlflow.set_experiment(experiment)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mlflow.langchain.autolog()      # traces LangGraph nodes
        except Exception:                   # pragma: no cover - langchain integration unavailable
            pass
        if llm.provider() != "mock":
            mlflow.openai.autolog()         # traces model calls, tokens and latency
