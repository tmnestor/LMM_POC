"""Generate a composite multi-receipt page PNG for transaction linking experiments.

Creates a single ~800x1800 image with 3 receipts stacked vertically on a grey
page background, with slight rotation to simulate stapled receipts scanned together.

Receipts:
  1. Office Supplies Plus — $83.48, 15/01/2024
  2. MetroCafe & Grill — $39.70, 12/01/2024
  3. Sydney Auto Parts Warehouse — $142.80, 10/01/2024

Run standalone: python experiments/synthetic/generate_multi_receipt_page.py
"""

from pathlib import Path

from PIL import Image, ImageDraw

from experiments.synthetic.generate_receipt import get_font

# Each receipt: (store_name, abn, address, phone, date, receipt_no, items, payment)
RECEIPTS: list[dict] = [
    {
        "store": "Office Supplies Plus",
        "abn": "51 824 753 556",
        "address": "123 Business Ave, Sydney NSW 2000",
        "phone": "(02) 9876 5432",
        "date": "15/01/2024",
        "receipt_no": "20240115-042",
        "items": [
            ("A4 Printer Paper (500 sheets)", 1, 24.99),
            ("Ink Cartridge - Black XL", 1, 45.50),
            ("USB-C Cable 1.5m", 1, 12.99),
        ],
        "payment": "EFTPOS",
        "card_last4": "7823",
    },
    {
        "store": "MetroCafe & Grill",
        "abn": "23 456 789 012",
        "address": "45 Pitt St, Sydney NSW 2000",
        "phone": "(02) 8765 4321",
        "date": "12/01/2024",
        "receipt_no": "20240112-118",
        "items": [
            ("Flat White - Large", 2, 5.50),
            ("Avocado Toast", 1, 18.90),
            ("Fresh Orange Juice", 1, 9.80),
            ("Banana Bread (toasted)", 1, 5.50),
        ],
        "payment": "VISA",
        "card_last4": "3491",
    },
    {
        "store": "Sydney Auto Parts Warehouse",
        "abn": "67 890 123 456",
        "address": "88 Parramatta Rd, Granville NSW 2142",
        "phone": "(02) 9632 1478",
        "date": "10/01/2024",
        "receipt_no": "20240110-307",
        "items": [
            ("Engine Oil 5W-30 5L", 1, 42.00),
            ("Oil Filter - Standard", 1, 18.50),
            ("Wiper Blades (pair)", 1, 34.90),
            ("Brake Fluid DOT4 1L", 1, 22.40),
            ("Microfibre Cloths (5pk)", 1, 25.00),
        ],
        "payment": "EFTPOS",
        "card_last4": "5562",
    },
]

# Rotation angles (degrees) per receipt to simulate stapled scan
ROTATIONS = [-1.5, 0.8, -0.5]


def _draw_single_receipt(receipt: dict) -> Image.Image:
    """Render a single receipt onto a PIL Image.

    Args:
        receipt: Receipt data dict from RECEIPTS constant.

    Returns:
        PIL Image of the rendered receipt.
    """
    width = 700
    # Estimate height based on item count
    estimated_height = 420 + len(receipt["items"]) * 32
    img = Image.new("RGB", (width, estimated_height), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    font_title = get_font(24, bold=True)
    font_heading = get_font(17, bold=True)
    font_body = get_font(15)
    font_small = get_font(12)

    # Store header
    y = 20
    draw.text(
        (width // 2, y),
        receipt["store"],
        fill="#000000",
        font=font_title,
        anchor="mt",
    )
    y += 32
    draw.text(
        (width // 2, y),
        f"ABN: {receipt['abn']}",
        fill="#444444",
        font=font_small,
        anchor="mt",
    )
    y += 20
    draw.text(
        (width // 2, y),
        receipt["address"],
        fill="#444444",
        font=font_small,
        anchor="mt",
    )
    y += 20
    draw.text(
        (width // 2, y),
        f"Phone: {receipt['phone']}",
        fill="#444444",
        font=font_small,
        anchor="mt",
    )

    # Separator
    y += 25
    draw.line([(50, y), (width - 50, y)], fill="#000000", width=2)

    # Date and receipt number
    y += 12
    draw.text((60, y), f"Date: {receipt['date']}", fill="#000000", font=font_body)
    draw.text(
        (width - 60, y),
        f"Receipt #: {receipt['receipt_no']}",
        fill="#000000",
        font=font_body,
        anchor="rt",
    )

    # Separator
    y += 28
    draw.line([(50, y), (width - 50, y)], fill="#CCCCCC", width=1)

    # Column headers
    y += 10
    draw.text((60, y), "Item", fill="#000000", font=font_heading)
    draw.text((430, y), "Qty", fill="#000000", font=font_heading, anchor="mt")
    draw.text((width - 60, y), "Amount", fill="#000000", font=font_heading, anchor="rt")

    y += 25
    draw.line([(50, y), (width - 50, y)], fill="#CCCCCC", width=1)

    # Line items
    y += 10
    for name, qty, price in receipt["items"]:
        draw.text((60, y), name, fill="#000000", font=font_body)
        draw.text((430, y), str(qty), fill="#000000", font=font_body, anchor="mt")
        draw.text(
            (width - 60, y),
            f"${price:.2f}",
            fill="#000000",
            font=font_body,
            anchor="rt",
        )
        y += 28

    # Separator before totals
    y += 5
    draw.line([(300, y), (width - 50, y)], fill="#000000", width=2)

    # Totals
    subtotal = sum(price for _, _, price in receipt["items"])
    gst = round(subtotal / 11, 2)

    y += 12
    draw.text((340, y), "Subtotal (incl. GST):", fill="#000000", font=font_body)
    draw.text(
        (width - 60, y),
        f"${subtotal:.2f}",
        fill="#000000",
        font=font_body,
        anchor="rt",
    )
    y += 24
    draw.text((340, y), "GST included:", fill="#666666", font=font_body)
    draw.text(
        (width - 60, y),
        f"${gst:.2f}",
        fill="#666666",
        font=font_body,
        anchor="rt",
    )

    # Bold total
    y += 28
    draw.line([(300, y), (width - 50, y)], fill="#000000", width=2)
    y += 8
    font_total = get_font(20, bold=True)
    draw.text((340, y), "TOTAL:", fill="#000000", font=font_total)
    draw.text(
        (width - 60, y),
        f"${subtotal:.2f}",
        fill="#000000",
        font=font_total,
        anchor="rt",
    )

    # Payment method
    y += 35
    draw.line([(50, y), (width - 50, y)], fill="#CCCCCC", width=1)
    y += 12
    draw.text(
        (width // 2, y),
        f"Payment: {receipt['payment']}",
        fill="#000000",
        font=font_body,
        anchor="mt",
    )
    y += 22
    draw.text(
        (width // 2, y),
        f"Card: **** **** **** {receipt['card_last4']}",
        fill="#666666",
        font=font_small,
        anchor="mt",
    )

    # Footer
    y += 28
    draw.text(
        (width // 2, y),
        "Thank you for your purchase!",
        fill="#888888",
        font=font_small,
        anchor="mt",
    )

    # Crop to actual content height
    final_height = y + 25
    return img.crop((0, 0, width, final_height))


def generate_multi_receipt_page(output_path: Path | None = None) -> Path:
    """Generate a composite page with multiple receipts stacked vertically.

    Args:
        output_path: Where to save the PNG.
            Defaults to experiments/data/synthetic_multi_receipt_page.png.

    Returns:
        Path to the generated image.
    """
    if output_path is None:
        output_path = (
            Path(__file__).parent.parent / "data" / "synthetic_multi_receipt_page.png"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Render individual receipts
    receipt_images = [_draw_single_receipt(r) for r in RECEIPTS]

    # Calculate page dimensions
    page_width = 800
    margin_top = 40
    margin_between = 30
    total_height = (
        margin_top
        + sum(img.height for img in receipt_images)
        + margin_between * (len(receipt_images) - 1)
        + 40  # bottom margin
    )

    # Grey page background
    page = Image.new("RGB", (page_width, total_height), "#D8D8D8")

    # Paste each receipt with slight rotation
    y_offset = margin_top
    for receipt_img, angle in zip(receipt_images, ROTATIONS, strict=True):
        rotated = receipt_img.rotate(
            angle, resample=Image.BICUBIC, expand=True, fillcolor="#D8D8D8"
        )
        x_offset = (page_width - rotated.width) // 2
        page.paste(rotated, (x_offset, y_offset))
        y_offset += receipt_img.height + margin_between

    page.save(output_path, "PNG")
    return output_path


if __name__ == "__main__":
    path = generate_multi_receipt_page()
    print(f"Multi-receipt page saved to: {path}")
