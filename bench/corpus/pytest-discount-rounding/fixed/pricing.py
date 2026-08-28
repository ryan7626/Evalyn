"""Integer-cent price helpers with the rounding regression removed."""


def apply_discount(cents: int, percent: int) -> int:
    """Apply a whole-number percentage without losing fractional cents early."""
    return cents - (cents * percent // 100)
