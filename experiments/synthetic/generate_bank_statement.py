"""Generate a synthetic bank statement PNG for transaction linking experiments.

Creates a realistic bank statement image (1200x800 landscape) with:
- Bank header and account details
- ~10 transactions (mix of debits and credits) with running balance
- Key matches (truncated merchant names test fuzzy matching):
  - "OFFICE SUPPLIES PLU" debit of $83.48 on 15/01/2024
  - "METRO CAFE AND GRI" debit of $45.20 on 12/01/2024 (& → AND)
  - "SYDNEY AUTO PARTS W" debit of $142.80 on 10/01/2024

Run standalone: python experiments/synthetic/generate_bank_statement.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def get_font(
    size: int, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a system font with cascading fallback."""
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SF-Pro.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


# Transaction data: (date, description, debit, credit)
# Debit = money out, Credit = money in. One value is None per row.
TRANSACTIONS: list[tuple[str, str, float | None, float | None]] = [
    ("02/01/2024", "SALARY DEPOSIT - J SMITH", None, 4250.00),
    ("03/01/2024", "RENT PAYMENT - 42 GEORGE ST", 1800.00, None),
    ("05/01/2024", "WOOLWORTHS TOWN HALL", 67.35, None),
    ("08/01/2024", "TRANSFER FROM SAVINGS", None, 500.00),
    ("10/01/2024", "SYDNEY AUTO PARTS W", 142.80, None),
    ("12/01/2024", "METRO CAFE AND GRI", 45.20, None),
    ("15/01/2024", "OFFICE SUPPLIES PLU", 83.48, None),
    ("18/01/2024", "NETFLIX.COM", 22.99, None),
    ("22/01/2024", "ATM WITHDRAWAL CBD", 200.00, None),
    ("25/01/2024", "MEDICARE REBATE", None, 38.50),
    ("28/01/2024", "OPTUS MOBILE PLAN", 65.00, None),
]

OPENING_BALANCE = 3_245.67


def generate_bank_statement(output_path: Path | None = None) -> Path:
    """Generate a synthetic bank statement image.

    Args:
        output_path: Where to save the PNG. Defaults to experiments/data/synthetic_bank_statement.png.

    Returns:
        Path to the generated image.
    """
    if output_path is None:
        output_path = (
            Path(__file__).parent.parent / "data" / "synthetic_bank_statement.png"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Canvas — landscape statement
    width, height = 1200, 800
    img = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    # Fonts
    font_bank = get_font(30, bold=True)
    font_heading = get_font(18, bold=True)
    font_body = get_font(16)
    font_small = get_font(13)

    # --- Bank header (yellow band) ---
    draw.rectangle([(0, 0), (width, 70)], fill="#FFC72C")
    draw.text((30, 20), "Commonwealth Bank", fill="#000000", font=font_bank)
    draw.text(
        (width - 30, 35),
        "Statement of Account",
        fill="#333333",
        font=font_heading,
        anchor="rt",
    )

    # --- Account details ---
    y = 90
    draw.text((30, y), "Account Name:", fill="#666666", font=font_small)
    draw.text((160, y), "J Smith Business Account", fill="#000000", font=font_body)
    y += 25
    draw.text((30, y), "BSB:", fill="#666666", font=font_small)
    draw.text((160, y), "062-000", fill="#000000", font=font_body)
    draw.text((350, y), "Account:", fill="#666666", font=font_small)
    draw.text((430, y), "1234 5678", fill="#000000", font=font_body)
    y += 25
    draw.text((30, y), "Period:", fill="#666666", font=font_small)
    draw.text((160, y), "01/01/2024 - 31/01/2024", fill="#000000", font=font_body)

    # --- Opening balance ---
    y += 35
    draw.line([(30, y), (width - 30, y)], fill="#000000", width=2)
    y += 8
    draw.text((30, y), "Opening Balance", fill="#000000", font=font_heading)
    draw.text(
        (width - 30, y),
        f"${OPENING_BALANCE:,.2f}",
        fill="#000000",
        font=font_heading,
        anchor="rt",
    )

    # --- Column headers ---
    y += 35
    draw.rectangle([(30, y), (width - 30, y + 28)], fill="#F0F0F0")
    col_date = 50
    col_desc = 180
    col_debit = 700
    col_credit = 870
    col_balance = width - 50
    draw.text((col_date, y + 5), "Date", fill="#000000", font=font_heading)
    draw.text((col_desc, y + 5), "Description", fill="#000000", font=font_heading)
    draw.text(
        (col_debit, y + 5), "Debit", fill="#000000", font=font_heading, anchor="rt"
    )
    draw.text(
        (col_credit, y + 5), "Credit", fill="#000000", font=font_heading, anchor="rt"
    )
    draw.text(
        (col_balance, y + 5), "Balance", fill="#000000", font=font_heading, anchor="rt"
    )

    # --- Transactions ---
    y += 32
    balance = OPENING_BALANCE
    row_height = 30

    for i, (date, desc, debit, credit) in enumerate(TRANSACTIONS):
        # Alternating row background
        if i % 2 == 0:
            draw.rectangle([(30, y), (width - 30, y + row_height)], fill="#FAFAFA")

        # Update balance
        if debit is not None:
            balance -= debit
        if credit is not None:
            balance += credit

        # Draw row
        draw.text((col_date, y + 6), date, fill="#000000", font=font_body)
        draw.text((col_desc, y + 6), desc, fill="#000000", font=font_body)
        if debit is not None:
            draw.text(
                (col_debit, y + 6),
                f"${debit:,.2f}",
                fill="#CC0000",
                font=font_body,
                anchor="rt",
            )
        if credit is not None:
            draw.text(
                (col_credit, y + 6),
                f"${credit:,.2f}",
                fill="#006600",
                font=font_body,
                anchor="rt",
            )
        draw.text(
            (col_balance, y + 6),
            f"${balance:,.2f}",
            fill="#000000",
            font=font_body,
            anchor="rt",
        )

        # Row separator
        y += row_height
        draw.line([(30, y), (width - 30, y)], fill="#E0E0E0", width=1)

    # --- Closing balance ---
    y += 15
    draw.line([(30, y), (width - 30, y)], fill="#000000", width=2)
    y += 8
    draw.text((30, y), "Closing Balance", fill="#000000", font=font_heading)
    draw.text(
        (width - 30, y),
        f"${balance:,.2f}",
        fill="#000000",
        font=font_heading,
        anchor="rt",
    )

    # --- Footer ---
    y += 40
    draw.text(
        (width // 2, y),
        "This is a computer-generated statement and does not require a signature.",
        fill="#999999",
        font=font_small,
        anchor="mt",
    )

    img.save(output_path, "PNG")
    return output_path


if __name__ == "__main__":
    path = generate_bank_statement()
    print(f"Bank statement saved to: {path}")
