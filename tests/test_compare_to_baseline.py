import inspect


def _manifest(call_types: dict) -> dict:
    return {"version": 1, "call_types": call_types, "defaults": {}}


def _ct(cases, n, baseline, tol, tier="text") -> dict:
    return {"tier": tier, "cases": cases, "n": n,
            "baseline_pass_rate": baseline, "tolerance": tol}


def _verdicts(n_pass, n_fail) -> list[dict]:
    return ([{"passed": True}] * n_pass) + ([{"passed": False}] * n_fail)


def test_no_regression_when_at_baseline(leerie):
    manifest = _manifest({"classifier": _ct(["c1"], 5, 0.8, 0.15)})
    # 4/5 = 0.8 == baseline → OK
    report = leerie.compare_to_baseline({"classifier": _verdicts(4, 1)}, manifest)
    assert report["overall"] == "OK"
    assert report["per_call_type"]["classifier"]["current"] == 0.8


def test_regression_below_tolerance(leerie):
    manifest = _manifest({"classifier": _ct(["c1"], 5, 0.8, 0.15)})
    # 3/5 = 0.6 < 0.8 - 0.15 = 0.65 → REGRESSED
    report = leerie.compare_to_baseline({"classifier": _verdicts(3, 2)}, manifest)
    assert report["overall"] == "REGRESSED"
    assert report["per_call_type"]["classifier"]["verdict"] == "REGRESSED"


def test_tolerance_edge_exactly_at_threshold_is_ok(leerie):
    manifest = _manifest({"classifier": _ct(["c1"], 4, 0.9, 0.15)})
    # 3/4 = 0.75; threshold 0.9 - 0.15 = 0.75; current < threshold is False → OK
    report = leerie.compare_to_baseline({"classifier": _verdicts(3, 1)}, manifest)
    assert report["overall"] == "OK"


def test_empty_corpus_is_ok_with_warning(leerie):
    report = leerie.compare_to_baseline({}, _manifest({}))
    assert report["overall"] == "OK"
    assert report["per_call_type"] == {}
    assert any("empty" in w.lower() for w in report["warnings"])


def test_total_zero_does_not_divide(leerie):
    # No verdicts supplied for a non-empty manifest entry → passes 0 / total 5.
    manifest = _manifest({"classifier": _ct(["c1"], 5, 0.8, 0.15)})
    report = leerie.compare_to_baseline({}, manifest)
    assert report["per_call_type"]["classifier"]["current"] == 0.0
    assert report["overall"] == "REGRESSED"


def test_multi_call_type_one_regresses_one_improves(leerie):
    manifest = _manifest({
        "classifier": _ct(["c1"], 5, 0.9, 0.10),   # 5/5=1.0 → OK (improved)
        "planner": _ct(["p1"], 5, 0.9, 0.10),       # 1/5=0.2 → REGRESSED
    })
    report = leerie.compare_to_baseline(
        {"classifier": _verdicts(5, 0), "planner": _verdicts(1, 4)}, manifest)
    assert report["overall"] == "REGRESSED"
    assert report["per_call_type"]["classifier"]["verdict"] == "OK"
    assert report["per_call_type"]["planner"]["verdict"] == "REGRESSED"


def test_env_tier_boundary_float_repr_is_ok(leerie):
    # 9/15 == 0.6 exactly, but 0.8 - 0.20 == 0.6000000000000001 in IEEE754, so
    # a naive `current < baseline - tolerance` spuriously REGRESSED (exit 12)
    # at the exact boundary. The epsilon-guarded predicate keeps it OK.
    manifest = _manifest(
        {"implementer": _ct(["i1", "i2", "i3"], 5, 0.8, 0.20, tier="env")})
    report = leerie.compare_to_baseline(
        {"implementer": _verdicts(9, 6)}, manifest)
    assert report["per_call_type"]["implementer"]["current"] == 0.6
    assert report["overall"] == "OK"


def test_regressed_semantics_match_check_convergence(leerie):
    """Coupling test: the comparator and the heal-loop convergence check must
    decide "below baseline" through the SAME predicate so they cannot drift.
    Both now route through `_regressed_below` — compare_to_baseline with the
    manifest tolerance, check_convergence's REGRESSED arm with tolerance 0.
    Asserts the shared call is present in both AND that the predicate is
    epsilon-safe at the exact boundary (behavioral, not a source-substring
    marker)."""
    comp_src = inspect.getsource(leerie.compare_to_baseline)
    conv_src = inspect.getsource(leerie.check_convergence)
    assert "_regressed_below(" in comp_src, (
        "compare_to_baseline must decide via the shared _regressed_below "
        "predicate")
    assert "_regressed_below(" in conv_src, (
        "check_convergence's REGRESSED arm must decide via _regressed_below")
    # Exactly-at-boundary is OK (not a regression), despite float rounding.
    assert leerie._regressed_below(0.6, 0.8, 0.20) is False
    # A genuine drop past tolerance regresses.
    assert leerie._regressed_below(8 / 15, 0.8, 0.20) is True
    # tolerance=0 (the convergence arm's usage): strictly-below-baseline only.
    assert leerie._regressed_below(0.8, 0.8, 0.0) is False
    assert leerie._regressed_below(0.79, 0.8, 0.0) is True
