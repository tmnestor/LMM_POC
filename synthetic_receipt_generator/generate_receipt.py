"""Generate a synthetic receipt PNG for transaction linking experiments.

Creates a realistic receipt image (800x600 portrait) with:
- Store name, ABN, date
- Itemised line items with prices
- Subtotal, GST, total
- Payment method

Run standalone: python experiments/synthetic/generate_receipt.py
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


def generate_receipt(output_path: Path | None = None) -> Path:
    """Generate a synthetic receipt image.

    Args:
        output_path: Where to save the PNG. Defaults to experiments/data/synthetic_receipt.png.

    Returns:
        Path to the generated image.
    """
    if output_path is None:
        output_path = Path(__file__).parent.parent / "data" / "synthetic_receipt.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Canvas — portrait receipt
    width, height = 800, 600
    img = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    # Fonts
    font_title = get_font(28, bold=True)
    font_heading = get_font(20, bold=True)
    font_body = get_font(18)
    font_small = get_font(14)

    # Store header
    y = 30
    draw.text(
        (width // 2, y),
        "Office Supplies Plus",
        fill="#000000",
        font=font_title,
        anchor="mt",
    )
    y += 40
    draw.text(
        (width // 2, y),
        "ABN: 51 824 753 556",
        fill="#444444",
        font=font_small,
        anchor="mt",
    )
    y += 25
    draw.text(
        (width // 2, y),
        "123 Business Ave, Sydney NSW 2000",
        fill="#444444",
        font=font_small,
        anchor="mt",
    )
    y += 25
    draw.text(
        (width // 2, y),
        "Phone: (02) 9876 5432",
        fill="#444444",
        font=font_small,
        anchor="mt",
    )

    # Separator
    y += 30
    draw.line([(60, y), (width - 60, y)], fill="#000000", width=2)

    # Date and receipt number
    y += 15
    draw.text((80, y), "Date: 15/01/2024", fill="#000000", font=font_body)
    draw.text(
        (width - 80, y),
        "Receipt #: 20240115-042",
        fill="#000000",
        font=font_body,
        anchor="rt",
    )

    # Separator
    y += 35
    draw.line([(60, y), (width - 60, y)], fill="#CCCCCC", width=1)

    # Column headers
    y += 12
    draw.text((80, y), "Item", fill="#000000", font=font_heading)
    draw.text((500, y), "Qty", fill="#000000", font=font_heading, anchor="mt")
    draw.text((width - 80, y), "Amount", fill="#000000", font=font_heading, anchor="rt")

    # Separator
    y += 30
    draw.line([(60, y), (width - 60, y)], fill="#CCCCCC", width=1)

    # Line items
    items = [
        ("A4 Printer Paper (500 sheets)", 1, 24.99),
        ("Ink Cartridge - Black XL", 1, 45.50),
        ("USB-C Cable 1.5m", 1, 12.99),
    ]
    y += 12
    for name, qty, price in items:
        draw.text((80, y), name, fill="#000000", font=font_body)
        draw.text((500, y), str(qty), fill="#000000", font=font_body, anchor="mt")
        draw.text(
            (width - 80, y),
            f"${price:.2f}",
            fill="#000000",
            font=font_body,
            anchor="rt",
        )
        y += 32

    # Separator before totals
    y += 5
    draw.line([(350, y), (width - 60, y)], fill="#000000", width=2)

    # Totals
    subtotal = sum(price for _, _, price in items)
    gst = round(subtotal / 11, 2)  # GST inclusive — GST = total / 11
    total = subtotal

    y += 15
    draw.text((400, y), "Subtotal (incl. GST):", fill="#000000", font=font_body)
    draw.text(
        (width - 80, y), f"${subtotal:.2f}", fill="#000000", font=font_body, anchor="rt"
    )
    y += 28
    draw.text((400, y), "GST included:", fill="#666666", font=font_body)
    draw.text(
        (width - 80, y), f"${gst:.2f}", fill="#666666", font=font_body, anchor="rt"
    )

    # Bold total
    y += 35
    draw.line([(350, y), (width - 60, y)], fill="#000000", width=2)
    y += 10
    font_total = get_font(24, bold=True)
    draw.text((400, y), "TOTAL:", fill="#000000", font=font_total)
    draw.text(
        (width - 80, y), f"${total:.2f}", fill="#000000", font=font_total, anchor="rt"
    )

    # Payment method
    y += 45
    draw.line([(60, y), (width - 60, y)], fill="#CCCCCC", width=1)
    y += 15
    draw.text(
        (width // 2, y), "Payment: EFTPOS", fill="#000000", font=font_body, anchor="mt"
    )
    y += 28
    draw.text(
        (width // 2, y),
        "Card: **** **** **** 7823",
        fill="#666666",
        font=font_small,
        anchor="mt",
    )

    # Footer
    y += 35
    draw.text(
        (width // 2, y),
        "Thank you for your purchase!",
        fill="#888888",
        font=font_small,
        anchor="mt",
    )

    img.save(output_path, "PNG")
    return output_path


if __name__ == "__main__":
    path = generate_receipt()
    print(f"Receipt saved to: {path}")
