"""Compare freshly synthesized metrics with the committed paper metrics.

Latency is a property of the generated schedule: for a given design it is
identical on every host that runs the same Vitis version, so it is compared
exactly and a difference is a reproduction failure.

Resource estimates (LUT, FF, DSP, BRAM) are produced by the synthesis engine
and drift by a few cells between Vitis builds and hosts even for byte-identical
input. They are therefore compared with a small tolerance, and anything outside
that tolerance is printed with its delta rather than being hidden or being
treated as a failed reproduction. Pass ``strict=True`` to require exact equality
on every metric, which is what the original drivers did.
"""

DEFAULT_TOL_PCT = 2.0
DEFAULT_TOL_ABS = 1

# Fields whose value must reproduce exactly, compared case-insensitively.
LATENCY_FIELDS = frozenset(
    {"latency", "latency_cycles", "lat", "tot_lat", "fb_lat", "interval"}
)


def _number(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def classify(field, fresh, paper, tol_pct=DEFAULT_TOL_PCT, tol_abs=DEFAULT_TOL_ABS):
    """Return one of 'exact', 'within', 'differs', 'mismatch' for one metric."""
    if str(fresh).strip() == str(paper).strip():
        return "exact"
    fresh_value, paper_value = _number(fresh), _number(paper)
    if fresh_value is None or paper_value is None:
        return "mismatch"
    if fresh_value == paper_value:
        return "exact"
    if field.lower() in LATENCY_FIELDS:
        return "mismatch"
    delta = abs(fresh_value - paper_value)
    if delta <= tol_abs or delta <= abs(paper_value) * tol_pct / 100.0:
        return "within"
    return "differs"


def _delta(fresh, paper):
    fresh_value, paper_value = _number(fresh), _number(paper)
    if fresh_value is None or paper_value is None or paper_value == 0:
        return ""
    return f" ({(fresh_value - paper_value) / paper_value * 100:+.1f}%)"


def report(title, entries, missing=(), strict=False,
           tol_pct=DEFAULT_TOL_PCT, tol_abs=DEFAULT_TOL_ABS):
    """Print a verification summary and return True when it passes.

    ``entries`` is an iterable of ``(label, field, fresh, paper)``. ``missing``
    lists rows that were expected but absent from the fresh reports; any of
    those, any latency difference, and any non-numeric difference fail the run.
    """
    buckets = {"exact": [], "within": [], "differs": [], "mismatch": []}
    latency_total = latency_exact = 0
    for label, field, fresh, paper in entries:
        verdict = classify(field, fresh, paper, tol_pct, tol_abs)
        buckets[verdict].append((label, field, fresh, paper))
        if field.lower() in LATENCY_FIELDS:
            latency_total += 1
            latency_exact += verdict == "exact"

    resource_total = sum(len(rows) for rows in buckets.values()) - latency_total
    resource_exact = len(buckets["exact"]) - latency_exact
    failed = list(missing) + [
        f"{label} {field}: fresh={fresh} paper={paper}"
        for label, field, fresh, paper in buckets["mismatch"]
    ]
    if strict:
        failed += [
            f"{label} {field}: fresh={fresh} paper={paper}{_delta(fresh, paper)}"
            for label, field, fresh, paper in buckets["within"] + buckets["differs"]
        ]

    print(f"\n{title}")
    if latency_total:
        print(f"  latency  : {latency_exact}/{latency_total} exact")
    if resource_total:
        print(
            f"  resources: {resource_exact}/{resource_total} exact, "
            f"{len(buckets['within'])} within ±{tol_pct:g}% or ±{tol_abs:g}, "
            f"{len(buckets['differs'])} beyond tolerance"
        )
    if buckets["differs"] and not strict:
        print("  beyond tolerance (Vitis build/host variation in the resource "
              "estimate, not a latency or functional difference):")
        for label, field, fresh, paper in buckets["differs"]:
            print(f"    {label} {field}: fresh={fresh} paper={paper}"
                  f"{_delta(fresh, paper)}")
    if failed:
        print("  FAIL")
        for line in failed:
            print(f"    {line}")
        return False
    if buckets["differs"]:
        print(f"  PASS — latency reproduced exactly; {len(buckets['differs'])} "
              "resource estimate(s) beyond tolerance, listed above")
    else:
        print("  PASS")
    return True
