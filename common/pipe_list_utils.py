"""
Utility functions for working with pipe-separated list strings.

This module provides a comprehensive set of functions for manipulating pipe-separated
lists (format: "item1 | item2 | item3"), commonly used in bank statement and document
extraction workflows.

Functions:
    - reverse_pipe_list(s): Reverse order of items
    - get_pipe_list_item(s, index): Get item at specific position
    - find_in_pipe_list(s, search): Find index of item (exact match)
    - find_all_by_values(s, searches): Find all indexes of items matching any search term (exact match)
    - find_all_by_substring(s, substring): Find all indexes containing substring (safe)
    - format_as_currency(s, negative, decimals, thousands_sep): Format as currency
    - pipe_list_length(s): Count number of items
    - delete_by_index(s, index): Delete item at position
    - delete_by_indexes(s, indexes): Delete multiple items by positions (safe batch delete)
    - delete_by_value(s, search): Delete first occurrence (exact match)
    - delete_all_by_value(s, search): Delete all occurrences (exact match)
    - delete_by_substring(s, substring): Delete all items containing substring (⚠️ dangerous)

Example:
    >>> from common.pipe_list_utils import reverse_pipe_list, format_as_currency
    >>> dates = "15/02/2024 | 13/02/2024 | 12/02/2024"
    >>> reverse_pipe_list(dates)
    '12/02/2024 | 13/02/2024 | 15/02/2024'
    >>> amounts = "4.49 | 23.0 | 25.5"
    >>> format_as_currency(amounts)
    '-$4.49 | -$23.00 | -$25.50'
"""


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


def find_all_by_values(s: str, searches: list[str]) -> list[int]:
    """
    Find all indexes of items that exactly match any of the search terms.

    Args:
        s: Pipe-separated string like "item1 | item2 | item3"
        searches: List of strings to search for (exact match)

    Returns:
        List of zero-based indexes where any search term was found (may include duplicates)

    Example:
        >>> find_all_by_values("apple | banana | cherry | banana", ["banana", "cherry"])
        [1, 2, 3]
        >>> find_all_by_values("apple | banana | cherry", ["orange", "grape"])
        []
        >>> find_all_by_values("red | blue | red | green", ["red"])
        [0, 2]
    """
    items = [item.strip() for item in s.split(" | ")]
    search_set = set(searches)
    return [i for i, item in enumerate(items) if item in search_set]


def format_as_currency(
    s: str, negative: bool = True, decimals: int = 2, thousands_sep: bool = False
) -> str:
    """
    Convert pipe-separated numeric values to formatted currency strings.

    Args:
        s: Pipe-separated numeric values like "4.49 | 23.0 | 25.5"
        negative: If True, prefix with "-$", otherwise just "$"
        decimals: Number of decimal places (default: 2)
        thousands_sep: If True, add comma separators for thousands (default: False)

    Returns:
        Pipe-separated currency strings like "-$4.49 | -$23.00 | -$25.50"

    Example:
        >>> format_as_currency("4.49 | 23.0 | 25.5")
        '-$4.49 | -$23.00 | -$25.50'
        >>> format_as_currency("4.49 | 23.0 | 25.5", negative=False)
        '$4.49 | $23.00 | $25.50'
        >>> format_as_currency("4.49 | 23.0 | 25.5", decimals=3)
        '-$4.490 | -$23.000 | -$25.500'
        >>> format_as_currency("3000.0 | 1500.5", thousands_sep=True)
        '-$3,000.00 | -$1,500.50'
    """
    items = s.split(" | ")
    prefix = "-$" if negative else "$"
    format_spec = f",.{decimals}f" if thousands_sep else f".{decimals}f"
    formatted = [f"{prefix}{float(item.strip()):{format_spec}}" for item in items]
    return " | ".join(formatted)


def pipe_list_length(s: str) -> int:
    """
    Calculate the number of items in a pipe-separated list.

    Args:
        s: Pipe-separated string like "item1 | item2 | item3"

    Returns:
        Number of items in the list

    Example:
        >>> pipe_list_length("apple | banana | cherry")
        3
        >>> pipe_list_length("single")
        1
        >>> pipe_list_length("")
        0
    """
    if not s or not s.strip():
        return 0
    return len(s.split(" | "))


def delete_by_index(s: str, index: int) -> str:
    """
    Delete item at specific index from pipe-separated list.

    Args:
        s: Pipe-separated string like "item1 | item2 | item3"
        index: Zero-based index of item to delete

    Returns:
        Pipe-separated string with item removed, or original string if index invalid

    Example:
        >>> delete_by_index("apple | banana | cherry", 1)
        'apple | cherry'
        >>> delete_by_index("apple | banana | cherry", 0)
        'banana | cherry'
        >>> delete_by_index("apple | banana | cherry", 5)
        'apple | banana | cherry'
    """
    items = s.split(" | ")
    if 0 <= index < len(items):
        items.pop(index)
        return " | ".join(items)
    return s


def delete_by_indexes(s: str, indexes: list[int]) -> str:
    """
    Delete multiple items by their positions from pipe-separated list.

    Items are removed in descending order to avoid index shifting issues.
    Invalid indexes are silently ignored.

    Args:
        s: Pipe-separated string like "item1 | item2 | item3"
        indexes: List of zero-based indexes to delete

    Returns:
        Pipe-separated string with specified items removed

    Example:
        >>> delete_by_indexes("apple | banana | cherry | date", [1, 3])
        'apple | cherry'
        >>> delete_by_indexes("apple | banana | cherry", [0, 2])
        'banana'
        >>> # Safe pattern with find_all_by_substring
        >>> data = "EFTPOS DEBIT | CARD DEBIT | ATM WITHDRAWAL"
        >>> matches = find_all_by_substring(data, "DEBIT")
        >>> delete_by_indexes(data, matches)
        'ATM WITHDRAWAL'
    """
    items = s.split(" | ")
    # Sort descending to avoid index shifting issues during deletion
    for idx in sorted(set(indexes), reverse=True):
        if 0 <= idx < len(items):
            items.pop(idx)
    return " | ".join(items)


def delete_by_value(s: str, search: str) -> str:
    """
    Delete first occurrence of search string from pipe-separated list.

    Args:
        s: Pipe-separated string like "item1 | item2 | item3"
        search: String to search for and remove

    Returns:
        Pipe-separated string with first matching item removed

    Example:
        >>> delete_by_value("apple | banana | cherry", "banana")
        'apple | cherry'
        >>> delete_by_value("apple | banana | apple", "apple")
        'banana | apple'
        >>> delete_by_value("apple | banana | cherry", "orange")
        'apple | banana | cherry'
    """
    items = [item.strip() for item in s.split(" | ")]
    try:
        items.remove(search)
        return " | ".join(items)
    except ValueError:
        return s


def delete_all_by_value(s: str, search: str) -> str:
    """
    Delete all occurrences of search string from pipe-separated list.

    Args:
        s: Pipe-separated string like "item1 | item2 | item3"
        search: String to search for and remove (all occurrences)

    Returns:
        Pipe-separated string with all matching items removed

    Example:
        >>> delete_all_by_value("apple | banana | apple | cherry", "apple")
        'banana | cherry'
        >>> delete_all_by_value("apple | banana | cherry", "banana")
        'apple | cherry'
        >>> delete_all_by_value("apple | banana | cherry", "orange")
        'apple | banana | cherry'
    """
    items = [item.strip() for item in s.split(" | ")]
    filtered = [item for item in items if item != search]
    return " | ".join(filtered)


def find_all_by_substring(s: str, substring: str) -> list[int]:
    """
    Find all indexes of items containing the substring.

    This is safer than delete_by_substring() as it lets you inspect matches
    before deletion using delete_by_indexes().

    Args:
        s: Pipe-separated string like "item1 | item2 | item3"
        substring: Substring to search for (case-sensitive)

    Returns:
        List of zero-based indexes where substring was found

    Example:
        >>> find_all_by_substring("apple pie | banana | cherry pie", "pie")
        [0, 2]
        >>> find_all_by_substring("EFTPOS DEBIT | CARD DEBIT | ATM WITHDRAWAL", "DEBIT")
        [0, 1]
        >>> find_all_by_substring("apple | banana | cherry", "xyz")
        []
    """
    items = [item.strip() for item in s.split(" | ")]
    return [i for i, item in enumerate(items) if substring in item]


def delete_by_substring(s: str, substring: str) -> str:
    """
    Delete all items containing the substring from pipe-separated list.

    WARNING: This function uses substring matching which can be dangerous.
    Consider using find_all_by_substring() + delete_by_indexes() for safer deletion.

    Args:
        s: Pipe-separated string like "item1 | item2 | item3"
        substring: Substring to search for (case-sensitive)

    Returns:
        Pipe-separated string with all items containing substring removed

    Example:
        >>> delete_by_substring("apple pie | banana | cherry pie", "pie")
        'banana'
        >>> delete_by_substring("EFTPOS DEBIT | CARD DEBIT | ATM WITHDRAWAL", "DEBIT")
        'ATM WITHDRAWAL'
        >>> delete_by_substring("apple | banana | cherry", "xyz")
        'apple | banana | cherry'
    """
    items = [item.strip() for item in s.split(" | ")]
    filtered = [item for item in items if substring not in item]
    return " | ".join(filtered)
