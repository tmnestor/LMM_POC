import pandas as pd


def remove_duplicate_strings(text):
    """Remove duplicate strings separated by ' | ' while preserving order."""
    # Handle NaN and NOT_FOUND values
    if pd.isna(text):
        return text
    if text == 'NOT_FOUND':
        return text

    # Convert to string to check if it contains a separator
    text_str = str(text)
    if ' | ' not in text_str:
        return text

    # Split by separator
    parts = text_str.split(' | ')

    # Remove duplicates while preserving order
    unique_parts = list(dict.fromkeys(parts))

    # Rejoin with separator
    return ' | '.join(unique_parts)


def apply_deduplication(df):
    """Apply deduplication to all columns in a DataFrame."""
    for col in df.columns:
        df[col] = df[col].apply(remove_duplicate_strings)
    return df
