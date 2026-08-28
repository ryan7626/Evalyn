"""Integer-cent price helpers with a seeded rounding regression."""


def apply_discount(cents: int, percent: int) -> int:
    """Apply a whole-number percentage."""
    return round(cents * (1 - percent / 100))
