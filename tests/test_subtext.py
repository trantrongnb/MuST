"""Unit tests for the multi-granularity composition. No CLIP or GPU needed."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from must.subtext import (  # noqa: E402
    build_class_prompts,
    build_spans,
    compose_span,
    format_class_name,
    load_catalog,
    pattern_count,
    span_geometry,
    span_lengths,
)

FAILURES = []


def check(condition, message):
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        FAILURES.append(message)


def test_span_counts():
    print("test_span_counts")
    check(pattern_count(4, "pyramid") == 10, "pyramid over 4 sub-texts gives M=10")
    check(pattern_count(4, "atomic") == 4, "atomic gives M=4")
    check(pattern_count(4, "full") == 1, "full gives M=1")
    check(pattern_count(4, "atomic_full") == 5, "atomic_full gives M=5")
    check(pattern_count(4, "pairs") == 7, "pairs gives M=7")
    check(pattern_count(4, "global") == 1, "global gives M=1")
    check(pattern_count(3, "pyramid") == 6, "pyramid over 3 sub-texts gives M=6")
    check(pattern_count(5, "pyramid") == 15, "pyramid over 5 sub-texts gives M=15")


def test_spans_are_contiguous_and_ordered():
    print("test_spans_are_contiguous_and_ordered")
    spans = build_spans(4, "pyramid")
    check(
        all(tuple(range(s[0], s[0] + len(s))) == s for s in spans),
        "every span is a contiguous index run",
    )
    lengths = [len(s) for s in spans]
    check(lengths == sorted(lengths), "spans are ordered from short to long")
    check(spans[0] == (0,) and spans[-1] == (0, 1, 2, 3), "first is atomic, last is full")
    check(len(set(spans)) == len(spans), "no duplicate spans")


def test_atomic_prefix_is_stable():
    print("test_atomic_prefix_is_stable")
    # Ablations must compare against the same atomic queries in the same order,
    # so the pyramid has to start with exactly the atomic spans.
    atomic = build_spans(4, "atomic")
    pyramid = build_spans(4, "pyramid")
    check(pyramid[: len(atomic)] == atomic, "pyramid starts with the atomic spans")
    check(build_spans(4, "pairs")[: len(atomic)] == atomic, "pairs starts with atomic")


def test_compose_span():
    print("test_compose_span")
    check(compose_span(["nock arrow"]) == "nock arrow.", "single span gets a period")
    check(
        compose_span(["nock arrow", "draw bow"]) == "nock arrow. Then draw bow.",
        "two sub-texts are joined by ' Then '",
    )
    check(
        compose_span(["Draw bow.", "aim"]) == "Draw bow. Then aim.",
        "existing punctuation is not duplicated",
    )
    check(compose_span(["a", "", "b"]) == "a. Then b.", "empty parts are dropped")


def test_format_class_name():
    print("test_format_class_name")
    check(format_class_name("brush_hair") == "brush hair", "underscores become spaces")
    check(
        format_class_name("ApplyEyeMakeup") == "Apply Eye Makeup",
        "camel case is split",
    )


def test_real_catalog():
    print("test_real_catalog")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "data", "sub_texts", "ucf101_class_subtexts.json")
    names, catalog = load_catalog(path)
    check(len(names) == 101, "UCF101 catalog holds 101 classes")
    check(all(len(items) == 4 for items in catalog), "every class has 4 sub-texts")
    check(names == sorted(names), "class order is deterministic")

    prompts = build_class_prompts(names, catalog, "pyramid")
    check(all(len(items) == 10 for items in prompts), "each class yields 10 prompts")
    archery = prompts[names.index("Archery")]
    check(archery[0] == "nock arrow.", "first pyramid prompt is the first sub-text")
    check(
        archery[-1]
        == "nock arrow. Then draw bow. Then aim bow. Then release arrow.",
        "last pyramid prompt concatenates all four sub-texts",
    )
    check(len(set(archery)) == 10, "the 10 prompts of a class are distinct")
    check(
        span_lengths(4, "pyramid") == [1, 1, 1, 1, 2, 2, 2, 3, 3, 4],
        "span lengths line up with the prompt order",
    )

    named = build_class_prompts(names, catalog, "full", prepend_class_name=True)
    check(
        named[names.index("Archery")][0].startswith("Archery. "),
        "--prepend_class_name prefixes the readable class name",
    )
    globals_ = build_class_prompts(names, catalog, "global")
    check(
        globals_[names.index("Archery")] == ["a video of a person performing Archery"],
        "global mode falls back to the plain class prompt",
    )


def test_span_geometry_is_n_agnostic():
    print("test_span_geometry_is_n_agnostic")
    # The prior must hold for any catalog size, including N that does not
    # divide the frame count.
    for subtext_count in (2, 3, 4, 5, 6):
        geometry = span_geometry(subtext_count, "pyramid")
        check(
            len(geometry) == pattern_count(subtext_count, "pyramid"),
            f"N={subtext_count}: geometry has one entry per span",
        )
        check(
            all(0.0 <= c <= 1.0 and 0.0 < w <= 0.5 for c, w, _ in geometry),
            f"N={subtext_count}: centers in [0,1] and half-widths in (0,0.5]",
        )
        center, half_width, length = geometry[-1]
        check(
            abs(center - 0.5) < 1e-9 and abs(half_width - 0.5) < 1e-9,
            f"N={subtext_count}: widest span covers the whole clip -> prior vanishes",
        )
        check(
            length == subtext_count,
            f"N={subtext_count}: widest span has length N",
        )
        # Atomic spans must tile the timeline without gaps or overlaps.
        atomic = [(c, w) for c, w, l in geometry if l == 1]
        edges = sorted(c - w for c, w in atomic) + [max(c + w for c, w in atomic)]
        check(
            abs(edges[0]) < 1e-9 and abs(edges[-1] - 1.0) < 1e-9,
            f"N={subtext_count}: atomic spans tile [0,1] end to end",
        )


def test_span_geometry_matches_spans():
    print("test_span_geometry_matches_spans")
    geometry = span_geometry(4, "pyramid")
    spans = build_spans(4, "pyramid")
    check(len(geometry) == len(spans), "geometry aligns with the span list")
    for span, (center, half_width, length) in zip(spans, geometry):
        start = span[0]
        check_silent = (
            abs(center - (2 * start + len(span)) / 8) < 1e-9
            and abs(half_width - len(span) / 8) < 1e-9
            and length == len(span)
        )
        if not check_silent:
            check(False, f"span {span} geometry is wrong")
            return
    check(True, "every span maps to the interval its sub-texts cover")
    check(
        span_geometry(4, "global") == [(0.5, 0.5, 4)],
        "global mode covers the whole clip, so the prior is inactive",
    )


def test_degenerate_catalogs_do_not_crash():
    print("test_degenerate_catalogs_do_not_crash")
    # A mode may request spans longer than the catalog holds.
    for mode in ("global", "atomic", "full", "atomic_full", "pairs", "pyramid"):
        try:
            count = pattern_count(1, mode)
            check(count == 1, f"N=1 mode={mode} degenerates to a single query")
        except Exception as error:  # noqa: BLE001
            check(False, f"N=1 mode={mode} raised {type(error).__name__}: {error}")


def main():
    for test in (
        test_span_counts,
        test_spans_are_contiguous_and_ordered,
        test_atomic_prefix_is_stable,
        test_compose_span,
        test_format_class_name,
        test_real_catalog,
        test_span_geometry_is_n_agnostic,
        test_span_geometry_matches_spans,
        test_degenerate_catalogs_do_not_crash,
    ):
        test()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed")
        raise SystemExit(1)
    print("all sub-text checks passed")


if __name__ == "__main__":
    main()
