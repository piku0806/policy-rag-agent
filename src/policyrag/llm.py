"""LLM access.

LLM_PROVIDER:
  mock        deterministic, offline (extractive answers), used for tests and CI
  databricks  Databricks Model Serving endpoint (OpenAI-compatible API)
  azure       Azure OpenAI / Microsoft Foundry deployment
  openai      OpenAI API
"""
from __future__ import annotations

import os
from functools import lru_cache


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "mock").lower()


@lru_cache
def client_and_model():
    p = provider()
    if p == "databricks":
        from openai import OpenAI
        host = os.environ["DATABRICKS_HOST"].rstrip("/")
        return OpenAI(base_url=f"{host}/serving-endpoints", api_key=os.environ["DATABRICKS_TOKEN"]), \
            os.getenv("LLM_MODEL", "databricks-meta-llama-3-3-70b-instruct")
    if p == "azure":
        from openai import AzureOpenAI
        return AzureOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], api_key=os.environ["AZURE_OPENAI_API_KEY"],
                           api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")), \
            os.environ["AZURE_OPENAI_DEPLOYMENT"]
    if p == "openai":
        from openai import OpenAI
        return OpenAI(), os.getenv("LLM_MODEL", "gpt-4.1-mini")
    raise RuntimeError(f"No client for provider {p!r}")


def complete(system: str, user: str) -> str:
    client, model = client_and_model()
    resp = client.chat.completions.create(model=model, temperature=0,
                                          messages=[{"role": "system", "content": system},
                                                    {"role": "user", "content": user}])
    return resp.choices[0].message.content or ""
