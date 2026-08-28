"""Probe the integer-cent rounding contract exposed by the selected diff."""

from pricing import apply_discount


def test_discount_preserves_integer_cent_rounding() -> None:
    assert apply_discount(1999, 15) == 1700
