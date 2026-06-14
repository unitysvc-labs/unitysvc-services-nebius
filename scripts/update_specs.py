#!/usr/bin/env python3
"""
Template-based update_services.py for Nebius.

Yields model dictionaries that are rendered using Jinja2 templates.

Usage: python scripts/update_services.py
"""

import os
import re
import sys
from pathlib import Path
from typing import Iterator

import httpx

from unitysvc_sellers.model_data import ModelDataFetcher, ModelDataLookup
from unitysvc_sellers.template_populate import populate_from_iterator

# Provider Configuration
PROVIDER_NAME = "nebius"
PROVIDER_DISPLAY_NAME = "Nebius"
API_BASE_URL = "https://api.studio.nebius.com/v1/"
ENV_API_KEY_NAME = "NEBIUS_API_KEY"

SCRIPT_DIR = Path(__file__).parent


class ModelSource:
    """Fetches models and yields template dictionaries."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.data_fetcher = ModelDataFetcher()
        self.litellm_data = None

    def iter_models(self) -> Iterator[dict]:
        """Yield model dictionaries for template rendering."""
        # Fetch LiteLLM data once
        self.litellm_data = self.data_fetcher.fetch_litellm_model_data()

        print(f"Fetching models from {PROVIDER_DISPLAY_NAME} API...")
        try:
            r = httpx.get(
                f"{API_BASE_URL}models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0,
            )
            r.raise_for_status()
            models = r.json().get("data", [])
            print(f"Found {len(models)} models\n")
        except Exception as e:
            print(f"Error listing models: {e}")
            return

        for i, model_info in enumerate(models, 1):
            model_id = model_info.get("id", "")
            print(f"[{i}/{len(models)}] {model_id}")

            # Build template variables
            template_vars = self._build_template_vars(model_id, model_info)
            if template_vars:
                yield template_vars
                print("  OK")

    def _build_template_vars(self, model_id: str, model_info: dict) -> dict:
        """Build template variables for a model."""
        service_type = self._determine_service_type(model_id)
        capabilities = self._determine_capabilities(service_type, model_id)
        is_vision = "vision" in capabilities
        display_name = model_id.replace("-", " ").replace("_", " ").title()

        # Build details from LiteLLM data and model info
        details = {}
        model_data = ModelDataLookup.lookup_model_details(
            model_id, self.litellm_data or {})

        if model_data:
            for field in [
                    "max_tokens", "max_input_tokens", "max_output_tokens",
                    "mode"
            ]:
                if field in model_data:
                    details[field] = model_data[field]
            if "litellm_provider" in model_data:
                details["litellm_provider"] = model_data["litellm_provider"]

        # Tool-call support per model.  Nebius's /v1/models doesn't expose
        # this, so we trust LiteLLM's model registry — but with two
        # corrections:
        #
        # 1. Prefer the ``nebius/<model_id>`` entry over arbitrary other
        #    provider rows.  ``ModelDataLookup`` matches the first
        #    ``*/<model_id>`` it finds in dict-iteration order, which can
        #    return e.g. the deepinfra row when nebius hosts the same
        #    model with different capabilities.
        # 2. A small denylist for models LiteLLM marks as tool-capable
        #    that Nebius nonetheless rejects with 400.  LiteLLM's per-
        #    provider rows are sometimes optimistic; this list captures
        #    upstream truth as we discover it.  Drop entries when Nebius
        #    actually adds tool support.
        nebius_specific = (self.litellm_data or {}).get(
            f"nebius/{model_id}", model_data
        )
        supports_function_calling = bool(
            (nebius_specific or {}).get("supports_function_calling")
        )
        if model_id in self._FC_DENYLIST:
            supports_function_calling = False

        if "owned_by" in model_info:
            details["owned_by"] = model_info["owned_by"]
        if "object" in model_info:
            details["object"] = model_info["object"]

        # Canonical (snake_case) metadata required by the platform validator
        # for LLM offerings.  Both keys must be present; null asserts
        # "unknown".  Claude models are closed-source so parameter_count
        # is permanently null per the canonical helper.  metadata_sources
        # records provenance so reviewers can triage stale-value reports.
        canonical = ModelDataLookup.get_canonical_metadata(
            model_id,
            fetcher=self.data_fetcher,
        )
        details["context_length"] = canonical["context_length"]
        details["parameter_count"] = canonical["parameter_count"]
        if canonical["sources"]:
            details["metadata_sources"] = canonical["sources"]

        # Extract pricing
        pricing = None
        if model_data:
            if "input_cost_per_token" in model_data and "output_cost_per_token" in model_data:
                input_price = float(
                    model_data["input_cost_per_token"]) * 1_000_000
                output_price = float(
                    model_data["output_cost_per_token"]) * 1_000_000
                pricing = {
                    "type": "one_million_tokens",
                    "input": self._format_price(input_price),
                    "output": self._format_price(output_price),
                    "description": "Pricing Per 1M Tokens Input/Output",
                    "reference": None,
                }

        # Description suffix tracks the offering's actual nature so
        # embeddings / VL / rerank don't all get described as "language
        # model".  VL is a sub-flavour of llm (capabilities tags it), so
        # check the vision capability before falling back to service_type.
        if is_vision:
            description_suffix = "vision-language model"
        else:
            description_suffix = {
                "llm": "language model",
                "embedding": "embedding model",
                "rerank": "reranker",
            }.get(service_type, "model")

        return {
            # Directory name uses -byok suffix (used by populate_from_iterator)
            "name": f"{model_id}-byok",
            # Offering name is the model_id (without -byok suffix)
            "offering_name": model_id,
            # Offering fields
            "display_name": display_name,
            "description": f"{display_name} {description_suffix}",
            "service_type": service_type,
            "capabilities": capabilities,
            "is_vision": is_vision,
            "status": "ready",
            "details": details,
            "payout_price": pricing,
            # Listing fields
            "list_price": pricing,
            "supports_function_calling": supports_function_calling,
            # Provider config (for templates)
            "provider_name": PROVIDER_NAME,
            "provider_display_name": PROVIDER_DISPLAY_NAME,
            "api_base_url": API_BASE_URL,
            "env_api_key_name": ENV_API_KEY_NAME,
        }

    # Vision-language model detection.  Most VL releases use the compact
    # ``VL`` suffix (``Qwen2.5-VL-72B``, ``InternVL``) or family names
    # like LLaVA / VLM that don't contain the word "vision" at all.  The
    # regex uses segment boundaries (``-`` / ``_`` / ``/`` / start / end)
    # so we don't false-positive on names that happen to contain "vl" as
    # a substring (e.g. "evaluator").  ``v\d+`` is intentionally NOT in
    # the alternation — it would catch version suffixes like
    # ``-v1`` / ``-v2`` and misclassify regular LLMs (e.g.
    # ``nvidia/Llama-3_1-Nemotron-Ultra-253B-v1``).  Single-letter ``V``
    # markers (GLM-4V) aren't covered yet — add an explicit per-model
    # entry if such a model lands in Nebius's catalog.
    _VL_PATTERN = re.compile(
        r"(?:^|[-_/])(?:vision|llava|vlm|vl)(?:[-_/]|\d+|$)",
        re.IGNORECASE,
    )

    # Models LiteLLM marks as tool-capable but Nebius's chat-completion
    # endpoint rejects with 400 when ``tools`` is sent.  Empirically
    # discovered — drop entries when Nebius adds upstream support.
    _FC_DENYLIST = frozenset({
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    })

    def _determine_service_type(self, model_id: str) -> str:
        """Map model id to a platform-recognised ``service_type``.

        The schema enum tracks *transport shape* (chat-completion vs
        embeddings vs rerank vs image-gen vs …), not per-feature flags.
        Vision-language models POST to the same ``/chat/completions``
        surface as text LLMs, so they stay ``llm`` here and pick up a
        ``"vision"`` tag in :meth:`_determine_capabilities` instead.
        """
        model_lower = model_id.lower()
        # Embeddings first — some embedding models also use "instruct" in
        # their name (e.g. Qwen3-Embedding-8B-Instruct), so the embedding
        # check has to win against any future LLM-leaning heuristic.
        if any(kw in model_lower for kw in ("embed", "embedding")):
            return "embedding"
        if "rerank" in model_lower:
            return "rerank"
        return "llm"

    def _determine_capabilities(self, service_type: str, model_id: str) -> list[str]:
        """Per-feature tags that ride alongside ``service_type``.

        The platform's capability taxonomy is open; we mirror what other
        provider repos emit so dashboards / search can group by feature.
        Vision detection reuses the same segment-aware regex as the old
        VL-as-service-type path — it catches ``Qwen2.5-VL`` /
        ``InternVL`` / ``LLaVA`` / explicit ``vision`` / ``vlm`` markers
        without false-positiving on ``-v1`` version suffixes.
        """
        caps: list[str] = [service_type]
        if service_type == "llm" and self._VL_PATTERN.search(model_id):
            caps.append("vision")
        return caps

    def _format_price(self, price: float) -> str:
        """Format price without trailing .0 for whole numbers."""
        if price == int(price):
            return str(int(price))
        return str(price)


def main():
    api_key = os.environ.get(ENV_API_KEY_NAME)
    if not api_key:
        print(f"Error: {ENV_API_KEY_NAME} not set")
        sys.exit(1)

    source = ModelSource(api_key)
    populate_from_iterator(
        iterator=source.iter_models(),
        templates_dir=SCRIPT_DIR.parent / "templates",
        output_dir=SCRIPT_DIR.parent / "services",
    )


if __name__ == "__main__":
    main()
