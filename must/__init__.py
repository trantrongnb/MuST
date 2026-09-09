"""MuST: Multi-granularity Sub-Text queries for few-shot action recognition.

Kept deliberately light: importing the package pulls in only the sub-text
machinery, so text-only tools (``scripts/check_subtext_length.py``,
``scripts/plot_span_prior.py``) do not pay for loading torch and transformers.
Import the model explicitly::

    from must.model import MuST
"""

from .subtext import (
    GRANULARITY_MODES,
    build_class_prompts,
    build_spans,
    compose_span,
    load_catalog,
    pattern_count,
    span_geometry,
    span_lengths,
)

__all__ = [
    "GRANULARITY_MODES",
    "build_class_prompts",
    "build_spans",
    "compose_span",
    "load_catalog",
    "pattern_count",
    "span_geometry",
    "span_lengths",
]
