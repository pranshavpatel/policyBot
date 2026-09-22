from rag.groundedness import check_groundedness


def test_grounded_answer_passes():
    ctx = "Employees receive 10 paid sick days annually."
    ans = "You receive 10 paid sick days annually."
    result = check_groundedness(ans, ctx)
    assert result.grounded is True
    assert result.coverage == 1.0
    assert result.unsupported_claims == []


def test_hallucinated_number_fails():
    ctx = "Employees receive 10 paid sick days annually."
    ans = "You receive 15 paid sick days annually."
    result = check_groundedness(ans, ctx)
    assert result.grounded is False
    assert "15" in result.unsupported_claims


def test_no_numeric_claims_defaults_to_grounded():
    ctx = "Employees may request unpaid leave for personal emergencies."
    ans = "You can request unpaid leave for a personal emergency."
    result = check_groundedness(ans, ctx)
    assert result.grounded is True
    assert result.coverage == 1.0


def test_partial_coverage_below_threshold_fails():
    ctx = "PTO carries over up to 5 days. Sick leave is 10 days."
    ans = "You can carry over 5 days and you get 25 sick days."
    result = check_groundedness(ans, ctx, min_coverage=1.0)
    assert result.grounded is False
    assert 0.0 < result.coverage < 1.0
