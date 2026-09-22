from tools.holiday_check import check_holiday, list_holidays, next_holidays


def test_check_holiday_true():
    out = check_holiday("2025-07-04")
    assert out["is_holiday"] is True
    assert out["name"] == "Independence Day"


def test_check_holiday_false():
    out = check_holiday("2025-07-05")
    assert out["is_holiday"] is False
    assert out["name"] is None


def test_check_holiday_invalid_format():
    import pytest
    with pytest.raises(ValueError):
        check_holiday("not-a-date")


def test_list_holidays_sorted_and_dated():
    out = list_holidays(2025)
    assert all(h["date"].startswith("2025-") for h in out)
    dates = [h["date"] for h in out]
    assert dates == sorted(dates)


def test_next_holidays_only_future_or_equal():
    out = next_holidays(n=3, start_date="2025-06-01")
    assert len(out) == 3
    for h in out:
        assert h["date"] >= "2025-06-01"


def test_next_holidays_crosses_year_boundary():
    # Dec 25 is the last fixed holiday; asking right after it should reach
    # into next year's Jan 1 / Jan 15 rather than returning nothing.
    out = next_holidays(n=2, start_date="2025-12-26")
    assert out[0]["date"] == "2026-01-01"
