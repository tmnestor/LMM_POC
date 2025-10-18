"""Utility functions for working with pipe-separated list strings."""


def reverse_pipe_list(s: str) -> str:
    """
    Reverse order of pipe-separated items.

    Args:
        s: Pipe-separated string like "item1 | item2 | item3"

    Returns:
        Reversed pipe-separated string like "item3 | item2 | item1"

    Example:
        >>> reverse_pipe_list("first | second | third")
        'third | second | first'
    """
    return " | ".join(s.split(" | ")[::-1])


def get_pipe_list_item(s: str, index: int) -> str:
    """
    Extract item at specific position from pipe-separated string.

    Args:
        s: Pipe-separated string like "item1 | item2 | item3"
        index: Zero-based index of item to retrieve

    Returns:
        Item at specified index, or empty string if index out of range

    Example:
        >>> get_pipe_list_item("apple | banana | cherry", 1)
        'banana'
        >>> get_pipe_list_item("apple | banana | cherry", 5)
        ''
    """
    items = s.split(" | ")
    if 0 <= index < len(items):
        return items[index].strip()
    return ""


def find_in_pipe_list(s: str, search: str) -> int:
    """
    Find index of search string in pipe-separated list.

    Args:
        s: Pipe-separated string like "item1 | item2 | item3"
        search: String to search for

    Returns:
        Zero-based index of first occurrence, or -1 if not found

    Example:
        >>> find_in_pipe_list("apple | banana | cherry", "banana")
        1
        >>> find_in_pipe_list("apple | banana | cherry", "orange")
        -1
    """
    items = [item.strip() for item in s.split(" | ")]
    try:
        return items.index(search)
    except ValueError:
        return -1


def format_as_currency(s: str, negative: bool = True, decimals: int = 2) -> str:
    """
    Convert pipe-separated numeric values to formatted currency strings.

    Args:
        s: Pipe-separated numeric values like "4.49 | 23.0 | 25.5"
        negative: If True, prefix with "-$", otherwise just "$"
        decimals: Number of decimal places (default: 2)

    Returns:
        Pipe-separated currency strings like "-$4.49 | -$23.00 | -$25.50"

    Example:
        >>> format_as_currency("4.49 | 23.0 | 25.5")
        '-$4.49 | -$23.00 | -$25.50'
        >>> format_as_currency("4.49 | 23.0 | 25.5", negative=False)
        '$4.49 | $23.00 | $25.50'
        >>> format_as_currency("4.49 | 23.0 | 25.5", decimals=3)
        '-$4.490 | -$23.000 | -$25.500'
    """
    items = s.split(" | ")
    prefix = "-$" if negative else "$"
    formatted = [f"{prefix}{float(item.strip()):.{decimals}f}" for item in items]
    return " | ".join(formatted)
