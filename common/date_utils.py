"""Date utility functions for formatting and conversion."""

from datetime import datetime


def convert_dates_to_readable(date_string: str) -> str:
    """
    Convert pipe-separated DD/MM/YYYY dates to readable 'DD Mon YYYY' format.

    Args:
        date_string: Pipe-separated dates like "15/02/2024 | 13/02/2024 | 12/02/2024"

    Returns:
        Pipe-separated readable dates like "15 Feb 2024 | 13 Feb 2024 | 12 Feb 2024"

    Example:
        >>> convert_dates_to_readable("15/02/2024 | 13/02/2024")
        '15 Feb 2024 | 13 Feb 2024'
    """
    items = date_string.split(" | ")
    converted = [
        datetime.strptime(d.strip(), "%d/%m/%Y").strftime("%d %b %Y") for d in items
    ]
    return " | ".join(converted)
