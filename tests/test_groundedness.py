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


def test_citation_line_section_number_is_not_a_claim():
    # Regression test for a real false positive hit in a live qwen3:32b
    # eval run: a correct "$50/month" answer got blocked because its own
    # "Source: ... — 2. Mental Health Resources" citation line cited a
    # *different* section number than the one the fact came from — that
    # "2" isn't a factual claim, it's provenance metadata, and shouldn't
    # need to appear in the context to count as grounded.
    ctx = "Wellness Program > 1. Wellness Stipend\n\nEmployees receive a $50/month wellness stipend."
    ans = "The monthly wellness stipend is $50.\nSource: wellness_program.md — 2. Mental Health Resources"
    result = check_groundedness(ans, ctx)
    assert result.grounded is True
    assert result.unsupported_claims == []


def test_hallucinated_number_still_caught_even_with_citation_line():
    # The fix above must not accidentally exempt a real hallucination that
    # happens to sit next to a citation line.
    ctx = "Employees receive 10 paid sick days annually."
    ans = "You receive 15 paid sick days annually.\nSource: leave_policy.md — 5. Sick Leave"
    result = check_groundedness(ans, ctx)
    assert result.grounded is False
    assert "15" in result.unsupported_claims
    assert "5" not in result.unsupported_claims  # the section number, correctly ignored
