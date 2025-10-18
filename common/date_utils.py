"""Date utility functions for formatting and conversion."""

from datetime import datetime


def convert_dates_to_readable(date_string: str, short_year: bool = False) -> str:
    """
    Convert pipe-separated DD/MM/YYYY dates to readable format.

    Args:
        date_string: Pipe-separated dates like "15/02/2024 | 13/02/2024 | 12/02/2024"
        short_year: If True, use 2-digit year format (default: False)

    Returns:
        Pipe-separated readable dates like "15 Feb 2024 | 13 Feb 2024 | 12 Feb 2024"
        or "15 Feb 24 | 13 Feb 24 | 12 Feb 24" if short_year=True

    Example:
        >>> convert_dates_to_readable("15/02/2024 | 13/02/2024")
        '15 Feb 2024 | 13 Feb 2024'
        >>> convert_dates_to_readable("06/08/2024", short_year=True)
        '06 Aug 24'
    """
    items = date_string.split(" | ")
    format_str = "%d %b %y" if short_year else "%d %b %Y"
    converted = [
        datetime.strptime(d.strip(), "%d/%m/%Y").strftime(format_str) for d in items
    ]
    return " | ".join(converted)
