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


def display_dataframe_summary(df, df_name="DataFrame"):
    """Display summary statistics for a DataFrame."""
    print(f"📊 {df_name} Structure:")
    print("=" * 70)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print(f"\n📋 {df_name} Head (first 5 rows):")
    print("=" * 70)
    print(df.head().to_string(index=False))

    print("\n📊 Column Summary:")
    print("=" * 70)

    # Percentage filled (excluding 'NOT_FOUND')
    for col in df.columns:
        if col != 'image_name':
            non_empty = (df[col] != 'NOT_FOUND').sum()
            total = len(df)
            pct = (non_empty / total) * 100
            print(f"{col:30s}: {non_empty:3d}/{total:3d} filled ({pct:5.1f}%)")
