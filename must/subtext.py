"""Step 1-2: the multi-granularity sub-text pyramid.

Every class ships with ``N`` atomic sub-texts describing consecutive phases of
the action, produced offline by an LLM.  For ``Archery``::

    ["nock arrow", "draw bow", "aim bow", "release arrow"]

A single atomic phrase describes an instant, so a *bag* of such phrases is
invariant to their permutation: it cannot tell "draw bow then release" apart
from "release then draw".  Rather than bolting a temporal module onto the visual
side, MuST builds the temporal structure in language space -- every contiguous
span of sub-texts is concatenated back into one sentence and pushed through the
frozen CLIP text encoder.  Longer spans describe longer stretches of the action,
so the class ends up represented at ``N`` temporal granularities.

For ``N = 4`` the ``pyramid`` mode yields ``M = N(N+1)/2 = 10`` queries::

    length 1   s1        s2        s3        s4        4 atomic queries
    length 2   s1>s2     s2>s3     s3>s4               3 transition queries
    length 3   s1>s2>s3  s2>s3>s4                      2 phase-group queries
    length 4   s1>s2>s3>s4                             1 whole-action query

The construction is pure string concatenation in front of a frozen encoder, so
it adds **zero trainable parameters**.  It also hands Step 3 a temporal centre
and width per query for free -- see :func:`span_geometry`.
"""

import json
import os
import re
from typing import Dict, List, Sequence, Tuple

# Span lengths kept by each granularity mode.  ``None`` means "every length from
# 1 to N", i.e. the full pyramid; ``-1`` is a placeholder for "the whole
# sequence".  ``global`` is special-cased: it ignores the sub-texts entirely and
# falls back to the plain class-name prompt.
GRANULARITY_MODES = {
    "global": (),
    "atomic": (1,),
    "full": (-1,),
    "atomic_full": (1, -1),
    "pairs": (1, 2),
    "pyramid": None,
}

JOIN_TOKEN = " Then "


def _clean_subtext(text: str) -> str:
    """Strip markdown bold and enumeration prefixes left over by the LLM."""
    text = text.replace("**", "")
    text = re.sub(r"^\s*\d+[\.)]\s*", "", text)
    return " ".join(text.split())


def format_class_name(name: str) -> str:
    """``brush_hair`` -> ``brush hair``; ``ApplyEyeMakeup`` -> ``Apply Eye Makeup``."""
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    return " ".join(name.split())


def load_catalog(path: str) -> Tuple[List[str], List[List[str]]]:
    """Read ``{class_name: [sub-text, ...]}`` and return sorted names + texts.

    Sorting is what keeps a class bound to the same text queries across
    episodes, splits and runs, so it must not be made configurable.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Sub-text catalog does not exist: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        raw: Dict[str, List[str]] = json.load(handle)
    if not raw:
        raise ValueError(f"Sub-text catalog is empty: {path}")
    class_names = sorted(raw)
    catalog = [[_clean_subtext(text) for text in raw[name]] for name in class_names]
    counts = sorted({len(items) for items in catalog})
    if len(counts) != 1:
        raise ValueError(
            f"All classes must share the same sub-text count, found {counts} in {path}"
        )
    return class_names, catalog


def build_spans(subtext_count: int, mode: str) -> List[Tuple[int, ...]]:
    """Contiguous index spans selected by ``mode``, ordered short to long."""
    if mode not in GRANULARITY_MODES:
        raise ValueError(
            f"Unknown granularity '{mode}', expected one of {sorted(GRANULARITY_MODES)}"
        )
    if subtext_count < 1:
        raise ValueError(f"subtext_count must be positive, got {subtext_count}")

    if mode == "global":
        return []

    lengths = GRANULARITY_MODES[mode]
    if lengths is None:
        selected = list(range(1, subtext_count + 1))
    else:
        selected = sorted({subtext_count if l == -1 else l for l in lengths})

    # A mode may ask for spans longer than the catalog has (e.g. 'pairs' on a
    # catalog with a single sub-text); keep only the lengths that exist.
    selected = [length for length in selected if 1 <= length <= subtext_count]
    if not selected:
        raise ValueError(
            f"Mode '{mode}' selects no valid span length for {subtext_count} sub-texts"
        )

    spans: List[Tuple[int, ...]] = []
    for length in selected:
        for start in range(subtext_count - length + 1):
            spans.append(tuple(range(start, start + length)))
    return spans


def compose_span(parts: Sequence[str]) -> str:
    """Join consecutive sub-texts into one sentence readable by CLIP.

    The join is what makes the pyramid order-sensitive: "draw bow. Then release
    arrow." and "release arrow. Then draw bow." are different token sequences,
    so CLIP maps them to different embeddings.
    """
    sentences = []
    for part in parts:
        sentence = part.strip()
        if not sentence:
            continue
        if sentence[-1] not in ".!?":
            sentence = f"{sentence}."
        sentences.append(sentence)
    return JOIN_TOKEN.join(sentences)


def build_class_prompts(
    class_names: Sequence[str],
    catalog: Sequence[Sequence[str]],
    granularity: str,
    global_prompt_template: str = "a video of a person performing {}",
    prepend_class_name: bool = False,
) -> List[List[str]]:
    """Return the ``M`` composed text prompts of every class.

    ``granularity='global'`` falls back to the plain CLIP class prompt, which is
    the text-side baseline of the ablation table.
    """
    subtext_count = len(catalog[0])
    spans = build_spans(subtext_count, granularity)

    prompts: List[List[str]] = []
    for name, items in zip(class_names, catalog):
        readable = format_class_name(name)
        if granularity == "global":
            prompts.append([global_prompt_template.format(readable)])
            continue
        composed = []
        for span in spans:
            sentence = compose_span([items[index] for index in span])
            if prepend_class_name:
                sentence = f"{readable}. {sentence}"
            composed.append(sentence)
        prompts.append(composed)
    return prompts


def span_geometry(subtext_count: int, granularity: str) -> List[Tuple[float, float, int]]:
    """Temporal extent of every span, as ``(center, half_width, length)``.

    Because the atomic sub-texts describe ``N`` *consecutive* phases, sub-text
    ``k`` corresponds to the k-th slice of the action timeline.  A span covering
    sub-texts ``start .. start + L - 1`` therefore covers the normalised
    interval ``[start / N, (start + L) / N]``, stored here as centre and
    half-width.

    Coordinates are normalised to ``[0, 1]``, never frame indices, so the
    geometry holds for any ``N`` and any frame count ``T`` -- including when
    ``N`` does not divide ``T``.  The longest span covers the whole timeline
    (``half_width = 0.5``), so every frame falls inside it and the prior of
    Step 3 vanishes for that query by construction rather than by a special case.

    This mapping is free: it requires no learning and no extra data, and it only
    exists because the pyramid nests spans.  A bag of atomic sub-texts gives a
    centre but a constant width; a single global prompt gives neither.
    """
    if granularity == "global":
        # The plain class prompt describes the entire action: full coverage.
        return [(0.5, 0.5, subtext_count)]

    geometry: List[Tuple[float, float, int]] = []
    for span in build_spans(subtext_count, granularity):
        start, length = span[0], len(span)
        center = (2 * start + length) / (2 * subtext_count)
        half_width = length / (2 * subtext_count)
        geometry.append((center, half_width, length))
    return geometry


def span_lengths(subtext_count: int, granularity: str) -> List[int]:
    """Length of each span, aligned with :func:`build_class_prompts` order."""
    if granularity == "global":
        return [subtext_count]
    return [len(span) for span in build_spans(subtext_count, granularity)]


def pattern_count(subtext_count: int, granularity: str) -> int:
    """``M``: how many text queries a class gets under this granularity."""
    if granularity == "global":
        return 1
    return len(build_spans(subtext_count, granularity))
