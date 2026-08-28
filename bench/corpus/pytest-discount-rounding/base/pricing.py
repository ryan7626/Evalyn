"""Integer-cent price helpers."""


def apply_discount(cents: int, percent: int) -> int:
    """Apply a whole-number percentage without losing fractional cents early."""
    return cents - (cents * percent // 100)
