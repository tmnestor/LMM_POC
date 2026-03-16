import argparse
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def generate_payment_receipt(width=300, height=400):
    """
    Generate a smaller payment receipt that might be stapled on top of a main receipt.
    These are typically smaller and simpler, like credit card payment slips or proof of payment.

    Args:
        width: Width of the payment receipt (default smaller than full receipts)
        height: Height of the payment receipt (default smaller than full receipts)

    Returns:
        payment_receipt_img: PIL Image of the payment receipt
    """
    # Create a payment receipt image
    padding = 10
    receipt_img = Image.new(
        "RGB", (width + padding * 2, height + padding * 2), color=(255, 255, 255)
    )
    draw = ImageDraw.Draw(receipt_img)

    # Draw receipt outline with slight border
    receipt_color = (250, 250, 250)  # Slightly off-white for the receipt paper
    draw.rectangle(
        [(padding, padding), (width + padding, height + padding)],
        fill=receipt_color,
        outline=(240, 240, 240),
    )

    # Try to load a font
    try:
        # Try different common font paths
        font_paths = [
            "/Library/Fonts/Arial.ttf",  # macOS
            "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS alternative
            "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",  # Linux
            "C:\\Windows\\Fonts\\arial.ttf",  # Windows
            "/usr/share/fonts/truetype/freefont/FreeMono.ttf",  # Linux alternative
        ]

        header_font = None
        regular_font = None
        small_font = None

        for font_path in font_paths:
            if Path(font_path).exists():
                try:
                    header_font = ImageFont.truetype(font_path, 18)  # Smaller header
                    regular_font = ImageFont.truetype(font_path, 14)  # Smaller regular
                    small_font = ImageFont.truetype(font_path, 12)  # Smaller small
                    break
                except IOError:
                    continue

        # Fall back to default if no font found
        if header_font is None:
            header_font = ImageFont.load_default()
            regular_font = ImageFont.load_default()
            small_font = ImageFont.load_default()

    except Exception:
        # Fallback to default font
        header_font = ImageFont.load_default()
        regular_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Payment methods and card types
    payment_methods = ["CREDIT", "DEBIT", "VISA", "MASTERCARD", "AMEX", "EFTPOS"]
    payment_method = random.choice(payment_methods)

    # Generate random transaction data
    amount = round(random.uniform(5.99, 199.99), 2)

    # Random date in the last year
    days_ago = random.randint(0, 365)
    receipt_date = datetime.now() - timedelta(days=days_ago)
    date_str = receipt_date.strftime("%m/%d/%Y")
    time_str = f"{random.randint(8, 23)}:{random.randint(0, 59):02d}"

    # Generate a random transaction number
    transaction_num = f"#{random.randint(10000, 999999)}"

    # Calculate positions
    y_pos = 20 + padding
    x_center = (width + padding * 2) // 2

    # Draw payment header
    draw.text(
        (x_center, y_pos),
        "PAYMENT RECEIPT",
        fill=(0, 0, 0),
        font=header_font,
        anchor="mt",
    )
    y_pos += 25

    # Draw divider
    left_margin = 20 + padding
    right_margin = width + padding - 20
    draw.line([(left_margin, y_pos), (right_margin, y_pos)], fill=(0, 0, 0), width=1)
    y_pos += 15

    # Transaction info
    draw.text((left_margin + 10, y_pos), "DATE:", fill=(0, 0, 0), font=regular_font)
    draw.text(
        (right_margin - 20, y_pos),
        date_str,
        fill=(0, 0, 0),
        font=regular_font,
        anchor="rt",
    )
    y_pos += 20

    draw.text((left_margin + 10, y_pos), "TIME:", fill=(0, 0, 0), font=regular_font)
    draw.text(
        (right_margin - 20, y_pos),
        time_str,
        fill=(0, 0, 0),
        font=regular_font,
        anchor="rt",
    )
    y_pos += 20

    draw.text((left_margin + 10, y_pos), "TRANS ID:", fill=(0, 0, 0), font=regular_font)
    draw.text(
        (right_margin - 20, y_pos),
        transaction_num,
        fill=(0, 0, 0),
        font=regular_font,
        anchor="rt",
    )
    y_pos += 20

    # Payment method
    draw.text((left_margin + 10, y_pos), "METHOD:", fill=(0, 0, 0), font=regular_font)
    draw.text(
        (right_margin - 20, y_pos),
        payment_method,
        fill=(0, 0, 0),
        font=regular_font,
        anchor="rt",
    )
    y_pos += 30

    # Divider
    draw.line([(left_margin, y_pos), (right_margin, y_pos)], fill=(0, 0, 0), width=1)
    y_pos += 15

    # Amount
    draw.text((left_margin + 10, y_pos), "AMOUNT:", fill=(0, 0, 0), font=header_font)
    draw.text(
        (right_margin - 20, y_pos),
        f"${amount:.2f}",
        fill=(0, 0, 0),
        font=header_font,
        anchor="rt",
    )
    y_pos += 30

    # Add credit card info if applicable
    if payment_method in ["CREDIT", "VISA", "MASTERCARD", "AMEX"]:
        card_last_four = f"XXXX-XXXX-XXXX-{random.randint(1000, 9999)}"
        draw.text((left_margin + 10, y_pos), "CARD:", fill=(0, 0, 0), font=small_font)
        draw.text(
            (right_margin - 20, y_pos),
            card_last_four,
            fill=(0, 0, 0),
            font=small_font,
            anchor="rt",
        )
        y_pos += 20

        auth_code = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=6))
        draw.text((left_margin + 10, y_pos), "AUTH:", fill=(0, 0, 0), font=small_font)
        draw.text(
            (right_margin - 20, y_pos),
            auth_code,
            fill=(0, 0, 0),
            font=small_font,
            anchor="rt",
        )
        y_pos += 30

    # Status
    draw.text(
        (x_center, y_pos), "APPROVED", fill=(0, 0, 0), font=regular_font, anchor="mt"
    )
    y_pos += 25

    # Thank you message
    draw.text(
        (x_center, y_pos), "THANK YOU", fill=(0, 0, 0), font=small_font, anchor="mt"
    )

    # Add some realism effects

    # Add slight noise
    if random.random() < 0.7:
        noise_level = random.randint(1, 5)
        receipt_array = np.array(receipt_img)

        noise_mask = np.random.rand(*receipt_array.shape[:2]) < (noise_level / 100)
        for c in range(3):
            channel = receipt_array[:, :, c]
            noise = np.random.randint(0, 10, size=channel.shape)
            channel[noise_mask] = np.clip(
                channel[noise_mask] - noise[noise_mask], 0, 255
            )

        receipt_img = Image.fromarray(receipt_array)

    # Apply slight rotation
    if random.random() < 0.5:
        angle = random.uniform(-2, 2)
        receipt_img = receipt_img.rotate(
            angle, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255)
        )

    # Add staple marks (small dots in corners)
    if random.random() < 0.8:  # 80% chance to add staple marks
        staple_color = (120, 120, 120)  # Gray color for staple
        staple_positions = []

        # Choose 1-2 corners for staples
        num_staples = random.randint(1, 2)
        corners = [
            (padding + 5, padding + 5),  # Top-left
            (width + padding - 5, padding + 5),  # Top-right
            (padding + 5, height + padding - 5),  # Bottom-left
            (width + padding - 5, height + padding - 5),
        ]  # Bottom-right

        staple_positions = random.sample(corners, num_staples)

        for pos in staple_positions:
            # Draw a small circle or dot to represent the staple
            x, y = pos
            draw = ImageDraw.Draw(receipt_img)
            staple_size = 3
            draw.ellipse(
                [
                    (x - staple_size, y - staple_size),
                    (x + staple_size, y + staple_size),
                ],
                fill=staple_color,
            )

    return receipt_img


def generate_synthetic_receipt(width=500, height=800):
    """
    Generate a synthetic receipt from scratch with realistic text content and edges.

    Args:
        width: Width of the receipt
        height: Height of the receipt

    Returns:
        receipt_img: PIL Image of the synthetic receipt with realistic edges
    """
    # Create a slightly larger image to account for edge effects
    padding = 15
    receipt_img = Image.new(
        "RGB", (width + padding * 2, height + padding * 2), color=(255, 255, 255)
    )
    draw = ImageDraw.Draw(receipt_img)

    # Draw receipt outline with slight border
    receipt_color = (252, 252, 252)  # Very slight off-white for the receipt paper
    draw.rectangle(
        [(padding, padding), (width + padding, height + padding)],
        fill=receipt_color,
        outline=(240, 240, 240),
    )

    # Try to load Arial font, fall back to default if not available
    try:
        # Try different common font paths
        font_paths = [
            "/Library/Fonts/Arial.ttf",  # macOS
            "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS alternative
            "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",  # Linux
            "C:\\Windows\\Fonts\\arial.ttf",  # Windows
            "/usr/share/fonts/truetype/freefont/FreeMono.ttf",  # Linux alternative
        ]

        # Try to find a usable font
        header_font = None
        regular_font = None
        small_font = None

        for font_path in font_paths:
            if Path(font_path).exists():
                try:
                    header_font = ImageFont.truetype(font_path, 24)
                    regular_font = ImageFont.truetype(font_path, 18)
                    small_font = ImageFont.truetype(font_path, 14)
                    break
                except IOError:
                    continue

        # Fall back to default if no font found
        if header_font is None:
            header_font = ImageFont.load_default()
            regular_font = ImageFont.load_default()
            small_font = ImageFont.load_default()

    except Exception:
        # Fallback to default font
        header_font = ImageFont.load_default()
        regular_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Generate random receipt content
    store_names = [
        "QUICKMART",
        "GROCERY WORLD",
        "VALUESTORE",
        "MEGA SHOP",
        "DAILY NEEDS",
        "FRESH MARKET",
        "CITY MART",
        "MINI SHOP",
        "SUPER SAVE",
        "FOOD PLUS",
        "CORNER STORE",
        "EXPRESS MART",
    ]

    item_categories = [
        (
            "PRODUCE",
            [
                "Apples",
                "Bananas",
                "Tomatoes",
                "Lettuce",
                "Carrots",
                "Potatoes",
                "Onions",
                "Avocados",
            ],
        ),
        ("DAIRY", ["Milk", "Cheese", "Yogurt", "Butter", "Eggs", "Cream", "Ice Cream"]),
        (
            "BAKERY",
            ["Bread", "Bagels", "Muffins", "Cake", "Cookies", "Donuts", "Pastries"],
        ),
        ("MEAT", ["Chicken", "Beef", "Pork", "Fish", "Turkey", "Lamb", "Sausages"]),
        (
            "BEVERAGES",
            ["Coffee", "Tea", "Soda", "Juice", "Water", "Energy Drink", "Beer", "Wine"],
        ),
        (
            "SNACKS",
            ["Chips", "Pretzels", "Nuts", "Crackers", "Popcorn", "Candy", "Chocolate"],
        ),
        (
            "HOUSEHOLD",
            [
                "Paper Towels",
                "Toilet Paper",
                "Detergent",
                "Soap",
                "Cleaning Spray",
                "Trash Bags",
            ],
        ),
    ]

    payment_methods = [
        "CASH",
        "VISA",
        "MASTERCARD",
        "DEBIT",
        "AMEX",
        "DISCOVER",
        "GIFT CARD",
    ]

    # Generate random receipt data
    store_name = random.choice(store_names)
    store_address = f"{random.randint(100, 9999)} Main St, City, State {random.randint(10000, 99999)}"
    store_phone = f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}"

    # Random date in the last year
    days_ago = random.randint(0, 365)
    receipt_date = datetime.now() - timedelta(days=days_ago)
    date_str = receipt_date.strftime("%m/%d/%Y")
    time_str = f"{random.randint(8, 23)}:{random.randint(0, 59):02d}"

    # Generate a random receipt number
    receipt_num = f"#{random.randint(1000, 999999)}"

    # Cashier information
    cashier_names = [
        "Alex",
        "Jamie",
        "Casey",
        "Sam",
        "Taylor",
        "Jordan",
        "Morgan",
        "Riley",
        "Avery",
        "Quinn",
    ]
    cashier_name = random.choice(cashier_names)
    register_num = random.randint(1, 20)

    # Generate random items
    items = []
    total_amount = 0.0
    num_categories = random.randint(1, min(5, len(item_categories)))
    selected_categories = random.sample(item_categories, num_categories)

    for _category, category_items in selected_categories:
        num_items = random.randint(1, 4)
        category_selections = random.sample(
            category_items, min(num_items, len(category_items))
        )

        for item in category_selections:
            quantity = random.randint(1, 5) if random.random() < 0.2 else 1
            price = round(random.uniform(0.99, 29.99), 2)
            items.append((item, quantity, price, price * quantity))
            total_amount += price * quantity

    # Calculate tax and final total
    tax_rate = round(random.uniform(0.05, 0.11), 3)
    tax_amount = round(total_amount * tax_rate, 2)
    final_total = total_amount + tax_amount

    # Payment information
    payment_method = random.choice(payment_methods)

    # Start drawing the receipt - adjust for padding
    y_pos = 20 + padding  # Starting y position with padding
    x_center = (width + padding * 2) // 2  # Center position with padding

    # Store header
    draw.text(
        (x_center, y_pos), store_name, fill=(0, 0, 0), font=header_font, anchor="mt"
    )
    y_pos += 30

    # Store info
    draw.text(
        (x_center, y_pos), store_address, fill=(0, 0, 0), font=small_font, anchor="mt"
    )
    y_pos += 20
    draw.text(
        (x_center, y_pos), store_phone, fill=(0, 0, 0), font=small_font, anchor="mt"
    )
    y_pos += 20

    # Date and receipt info
    draw.text(
        (x_center, y_pos),
        f"Date: {date_str} Time: {time_str}",
        fill=(0, 0, 0),
        font=small_font,
        anchor="mt",
    )
    y_pos += 20
    draw.text(
        (x_center, y_pos),
        f"Receipt {receipt_num}",
        fill=(0, 0, 0),
        font=small_font,
        anchor="mt",
    )
    y_pos += 20
    draw.text(
        (x_center, y_pos),
        f"Cashier: {cashier_name} Register: {register_num}",
        fill=(0, 0, 0),
        font=small_font,
        anchor="mt",
    )
    y_pos += 30

    # Content area with margins
    left_margin = 20 + padding
    right_margin = width + padding - 20

    # Divider
    draw.line([(left_margin, y_pos), (right_margin, y_pos)], fill=(0, 0, 0), width=1)
    y_pos += 15

    # Column headers
    draw.text((left_margin + 10, y_pos), "ITEM", fill=(0, 0, 0), font=regular_font)
    draw.text((right_margin - 170, y_pos), "QTY", fill=(0, 0, 0), font=regular_font)
    draw.text((right_margin - 120, y_pos), "PRICE", fill=(0, 0, 0), font=regular_font)
    draw.text(
        (right_margin - 60, y_pos),
        "TOTAL",
        fill=(0, 0, 0),
        font=regular_font,
        anchor="lt",
    )
    y_pos += 20

    # Draw divider
    draw.line([(left_margin, y_pos), (right_margin, y_pos)], fill=(0, 0, 0), width=1)
    y_pos += 15

    # Draw items
    for item, quantity, price, item_total in items:
        draw.text((left_margin + 10, y_pos), item, fill=(0, 0, 0), font=regular_font)
        draw.text(
            (right_margin - 170, y_pos),
            str(quantity),
            fill=(0, 0, 0),
            font=regular_font,
        )
        draw.text(
            (right_margin - 120, y_pos),
            f"${price:.2f}",
            fill=(0, 0, 0),
            font=regular_font,
        )
        draw.text(
            (right_margin - 60, y_pos),
            f"${item_total:.2f}",
            fill=(0, 0, 0),
            font=regular_font,
            anchor="lt",
        )
        y_pos += 25

    # Bottom divider
    draw.line([(left_margin, y_pos), (right_margin, y_pos)], fill=(0, 0, 0), width=1)
    y_pos += 20

    # Subtotal
    draw.text(
        (right_margin - 170, y_pos), "Subtotal:", fill=(0, 0, 0), font=regular_font
    )
    draw.text(
        (right_margin - 60, y_pos),
        f"${total_amount:.2f}",
        fill=(0, 0, 0),
        font=regular_font,
        anchor="lt",
    )
    y_pos += 25

    # Tax
    draw.text(
        (right_margin - 170, y_pos),
        f"Tax ({tax_rate * 100:.1f}%):",
        fill=(0, 0, 0),
        font=regular_font,
    )
    draw.text(
        (right_margin - 60, y_pos),
        f"${tax_amount:.2f}",
        fill=(0, 0, 0),
        font=regular_font,
        anchor="lt",
    )
    y_pos += 25

    # Total
    draw.text((right_margin - 170, y_pos), "TOTAL:", fill=(0, 0, 0), font=header_font)
    draw.text(
        (right_margin - 60, y_pos),
        f"${final_total:.2f}",
        fill=(0, 0, 0),
        font=header_font,
        anchor="lt",
    )
    y_pos += 35

    # Payment method
    draw.text(
        (x_center, y_pos),
        f"PAID WITH {payment_method}",
        fill=(0, 0, 0),
        font=regular_font,
        anchor="mt",
    )
    y_pos += 30

    # Thank you message
    draw.text(
        (x_center, y_pos),
        "THANK YOU FOR SHOPPING WITH US",
        fill=(0, 0, 0),
        font=small_font,
        anchor="mt",
    )
    y_pos += 20
    draw.text(
        (x_center, y_pos),
        "PLEASE COME AGAIN",
        fill=(0, 0, 0),
        font=small_font,
        anchor="mt",
    )

    # Add some random noise/grain for realism
    if random.random() < 0.7:  # 70% chance to add noise
        noise_level = random.randint(1, 5)
        receipt_array = np.array(receipt_img)

        # Add noise to random pixels
        noise_mask = np.random.rand(*receipt_array.shape[:2]) < (noise_level / 100)
        for c in range(3):  # RGB channels
            channel = receipt_array[:, :, c]
            noise = np.random.randint(0, 10, size=channel.shape)
            channel[noise_mask] = np.clip(
                channel[noise_mask] - noise[noise_mask], 0, 255
            )

        receipt_img = Image.fromarray(receipt_array)

    # Apply slight rotation for realism
    if random.random() < 0.5:  # 50% chance to add rotation
        angle = random.uniform(-2, 2)  # Subtle rotation
        receipt_img = receipt_img.rotate(
            angle, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255)
        )

    # Add realistic receipt edge effects

    # 1. Simulate perforation at the top and/or bottom (dotted line pattern)
    if random.random() < 0.6:  # 60% chance to add perforation
        perf_width = width + padding * 2
        perforation = Image.new("RGB", (perf_width, padding), color=(255, 255, 255))
        perf_draw = ImageDraw.Draw(perforation)

        # Draw dotted line
        dot_spacing = 5
        dot_size = 2
        for x in range(0, perf_width, dot_spacing):
            # Draw small dots or rectangles
            perf_draw.rectangle(
                [
                    (x, padding // 2 - dot_size // 2),
                    (x + dot_size, padding // 2 + dot_size // 2),
                ],
                fill=(230, 230, 230),
            )

        # Add to top or bottom of receipt (sometimes both)
        if random.random() < 0.7:  # 70% chance for top perforation
            receipt_img.paste(perforation, (0, 0))
        if random.random() < 0.4:  # 40% chance for bottom perforation
            receipt_img.paste(perforation, (0, height + padding))

    # 2. Add slight edge irregularities
    if random.random() < 0.7:  # 70% chance to add edge irregularities
        receipt_array = np.array(receipt_img)

        # Left and right edges
        for y in range(receipt_array.shape[0]):
            # Left edge variation
            if random.random() < 0.1:  # Sparse effect
                left_margin = random.randint(0, 3)
                receipt_array[y, :left_margin, :] = 255  # White edge

            # Right edge variation
            if random.random() < 0.1:  # Sparse effect
                right_start = receipt_array.shape[1] - random.randint(1, 3)
                receipt_array[y, right_start:, :] = 255  # White edge

        receipt_img = Image.fromarray(receipt_array)

    # 3. Add subtle crease or fold marks (randomly)
    if random.random() < 0.4:  # 40% chance to add crease
        # Horizontal crease
        crease_y = random.randint(padding + height // 4, padding + 3 * height // 4)
        crease_width = 2
        crease_opacity = random.randint(5, 15)  # Very subtle

        for y in range(crease_y - crease_width, crease_y + crease_width):
            if 0 <= y < receipt_img.height:
                for x in range(0, receipt_img.width):
                    # Get current pixel
                    r, g, b = receipt_img.getpixel((x, y))
                    # Apply subtle darkening
                    r = max(0, r - crease_opacity)
                    g = max(0, g - crease_opacity)
                    b = max(0, b - crease_opacity)
                    receipt_img.putpixel((x, y), (r, g, b))

    # 4. Add shadow to bottom edge (thermal receipts often curl slightly)
    if random.random() < 0.5:  # 50% chance to add bottom shadow
        shadow_height = 4
        shadow_area = receipt_img.crop(
            (0, height + padding - shadow_height, width + padding * 2, height + padding)
        )
        shadow_array = np.array(shadow_area)

        # Gradually darken pixels
        for i in range(shadow_height):
            factor = (i + 1) / shadow_height * 10  # Gradual darkening
            shadow_array[i, :, :] = np.clip(shadow_array[i, :, :] - factor, 0, 255)

        shadow_area = Image.fromarray(shadow_array)
        receipt_img.paste(shadow_area, (0, height + padding - shadow_height))

    return receipt_img


def create_receipt_collage(
    canvas_size=(1600, 1200), receipt_count=None, stapled_ratio=0.0
):
    """
    Create a collage of synthetic receipts arranged centrally with minimal spacing.
    For 0-receipt examples, generates an Australian tax document in portrait orientation.

    Args:
        canvas_size: Size of the output canvas (width, height)
        receipt_count: Number of receipts to include
        stapled_ratio: Proportion of receipts with payment receipts stapled on top (0.0-1.0)

    Returns:
        collage_img: PIL Image of the collage or tax document for 0-receipt cases
        actual_count: Number of receipts in the collage

    Note:
        When receipt_count is 0, this function returns an Australian tax document (ATO notice,
        PAYG summary, etc.) in portrait orientation to represent real-world tax documents
        that might be included in tax submissions but aren't receipts.

        When stapled_ratio > 0, some receipts will have smaller payment receipts stapled
        on top, as commonly seen in tax documentation. This creates more realistic
        training data that mimics real-world receipts in tax submissions.
    """
    # Always create a plain white canvas
    canvas = Image.new("RGB", canvas_size, color=(255, 255, 255))

    # If we need 0 receipts, generate an Australian tax document in portrait orientation
    if receipt_count == 0:
        try:
            # Import tax document generator if we need it
            from create_tax_documents import generate_tax_document

            # Generate a standard A4 portrait tax document (595x842 pixels at 72 DPI)
            tax_document = generate_tax_document(width=595, height=842)

            # Calculate how to fit tax document into canvas while preserving portrait orientation
            tax_width, tax_height = tax_document.size
            canvas_width, canvas_height = canvas_size

            # Always ensure tax document is in portrait orientation (height > width)
            # Even if the collage itself is landscape or portrait
            if tax_height < tax_width:
                # Rotate if document is not in portrait orientation
                tax_document = tax_document.transpose(Image.ROTATE_90)
                tax_width, tax_height = tax_height, tax_width

            # Calculate scale to fit while maintaining portrait orientation
            scale = min(canvas_height / tax_height, canvas_width / tax_width)
            new_height = int(tax_height * scale)
            new_width = int(tax_width * scale)
            tax_document_resized = tax_document.resize(
                (new_width, new_height), Image.LANCZOS
            )

            # Center the resized tax document on the canvas
            paste_x = (canvas_width - new_width) // 2
            paste_y = (canvas_height - new_height) // 2
            canvas.paste(tax_document_resized, (paste_x, paste_y))

            # Return the canvas with properly oriented tax document
            return canvas, 0
        except (ImportError, Exception) as e:
            print(f"Warning: Could not generate tax document: {e}")
            # Fall back to empty canvas if tax document generation fails
            return canvas, 0

    # Use the full canvas with minimal margins
    # Calculate the center region (98% of the canvas)
    canvas_width, canvas_height = canvas_size
    center_margin_x = int(canvas_width * 0.01)  # 1% margin on left and right
    center_margin_y = int(canvas_height * 0.01)  # 1% margin on top and bottom

    # Define the usable area (central region)
    usable_width = canvas_width - 2 * center_margin_x
    usable_height = canvas_height - 2 * center_margin_y

    # Calculate optimal grid based on receipt count
    # For few receipts, use a tighter layout
    if receipt_count <= 2:
        # For 1-2 receipts, use a single row arrangement
        grid_rows = 1
        grid_columns = receipt_count
    elif receipt_count == 3:
        # For 3 receipts, use a single row
        grid_rows = 1
        grid_columns = 3
    elif receipt_count == 4:
        # For 4 receipts, use a 2x2 grid
        grid_rows = 2
        grid_columns = 2
    elif receipt_count == 5:
        # For exactly 5 receipts, arrange optimally in a more balanced pattern
        # Use a more compact arrangement with 3 on top, 2 on bottom
        grid_rows = 2
        grid_columns = 3
    else:
        # For 6+ receipts, use a more compact arrangement
        grid_rows = math.ceil(math.sqrt(receipt_count))
        grid_columns = math.ceil(receipt_count / grid_rows)

    # Calculate cell dimensions with NO spacing
    spacing = 0  # Zero spacing between receipts
    cell_width = usable_width // grid_columns
    cell_height = usable_height // grid_rows

    # Calculate starting position (top-left of usable area)
    start_x = center_margin_x
    start_y = center_margin_y

    # For really small receipt counts (1-2), make them appear larger
    if receipt_count <= 2:
        # Scale up individual receipts to use 100% of the canvas
        cell_width = int(usable_width / receipt_count)  # Use 100% of usable width
        cell_height = usable_height  # Use 100% of usable height
        # Recenter
        start_x = center_margin_x
        start_y = center_margin_y

    # Placement function that arranges receipts in the center area with minimal spacing
    def get_placement_position(index):
        """Get the position for a receipt based on its index"""
        if receipt_count == 0:
            return (0, 0, cell_width, cell_height)

        # Handle the special case of 5 receipts - center the bottom row
        if receipt_count == 5:
            # First three receipts in the top row
            if index < 3:
                row = 0
                col = index
            # Last two receipts centered in the bottom row
            else:
                row = 1
                # Center the bottom two receipts - offset for column calculation
                bottom_row_start = (
                    grid_columns - 2
                ) / 2  # Center point for 2 receipts in a 3-column grid
                col = int(bottom_row_start) + (index - 3)
        else:
            # Normal grid calculation for all other cases
            row = index // grid_columns
            col = index % grid_columns

        # Calculate cell position
        cell_x = start_x + col * cell_width
        cell_y = start_y + row * cell_height

        # Extremely minimal jitter for more consistent placement
        max_jitter = (
            min(cell_width, cell_height) // 40
        )  # Extremely small jitter (2.5% of cell)
        offset_x = random.randint(-max_jitter, max_jitter) if max_jitter > 0 else 0
        offset_y = random.randint(-max_jitter, max_jitter) if max_jitter > 0 else 0

        # Adjust position to center in cell with minimal jitter
        x = cell_x + offset_x
        y = cell_y + offset_y

        return (x, y, cell_width, cell_height)

    # Place each receipt
    actual_count = 0

    # Generate and place receipts
    for i in range(receipt_count):
        try:
            # Get position for this receipt
            x, y, cell_width, cell_height = get_placement_position(i)

            # Generate a synthetic receipt with dimensions optimized for this layout
            # Make receipt size more consistent for cleaner arrangement
            # Use larger receipts for zoom effect
            base_width = 400  # Wider base width for alignment
            height_variation = random.randint(
                -30, 70
            )  # Limited height variation for consistency

            if receipt_count <= 2:
                # For 1-2 receipts, make them much larger (max zoom effect)
                receipt_width = int(base_width * 1.4)  # 40% wider
                receipt_height = 900 + height_variation  # Much taller
            elif receipt_count <= 4:
                # For 3-4 receipts, still make them larger but fit the grid
                receipt_width = int(base_width * 1.2)  # 20% wider
                receipt_height = 800 + height_variation
            else:
                # For more receipts, make them slightly more compact but still large
                receipt_width = base_width
                receipt_height = 750 + height_variation

            receipt = generate_synthetic_receipt(
                width=receipt_width, height=receipt_height
            )

            # Use full cell with no margin
            margin = 0  # No margin
            max_width = cell_width
            max_height = cell_height

            # Calculate scale to fit within cell
            receipt_width, receipt_height = receipt.size
            width_scale = max_width / receipt_width
            height_scale = max_height / receipt_height
            scale = min(width_scale, height_scale)

            # Scale receipt to fit in cell
            new_width = int(receipt_width * scale)
            new_height = int(receipt_height * scale)
            receipt = receipt.resize((new_width, new_height), Image.LANCZOS)

            # Apply minimal rotation for realism - extremely slight for cleaner arrangement
            angle = random.uniform(-2, 2)  # Very minimal rotation
            receipt = receipt.rotate(
                angle, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255)
            )

            # Center the receipt in its cell
            cell_center_x = x + cell_width // 2
            cell_center_y = y + cell_height // 2

            # Calculate final position - centered in cell
            paste_x = cell_center_x - receipt.width // 2
            paste_y = cell_center_y - receipt.height // 2

            # Ensure receipt stays within canvas bounds
            paste_x = max(0, min(paste_x, canvas_size[0] - receipt.width))
            paste_y = max(0, min(paste_y, canvas_size[1] - receipt.height))

            # Paste the receipt onto the canvas
            canvas.paste(receipt, (paste_x, paste_y))
            actual_count += 1

            # Add stapled payment receipt on top of this receipt based on probability
            if random.random() < stapled_ratio:
                try:
                    # Create a smaller payment receipt
                    payment_width = int(
                        receipt.width * 0.6
                    )  # Payment receipt is ~60% of main receipt width
                    payment_height = int(
                        receipt.height * 0.5
                    )  # Payment receipt is ~50% of main receipt height
                    payment_receipt = generate_payment_receipt(
                        width=int(payment_width * 0.8),  # Account for padding
                        height=int(payment_height * 0.8),  # Account for padding
                    )

                    # Apply slight rotation for realism
                    angle = random.uniform(-5, 5)
                    payment_receipt = payment_receipt.rotate(
                        angle,
                        expand=True,
                        resample=Image.BICUBIC,
                        fillcolor=(255, 255, 255),
                    )

                    # Position the payment receipt to partially overlap the main receipt
                    # (typically in the top portion)
                    overlap_x = random.randint(
                        paste_x, paste_x + receipt.width - payment_receipt.width
                    )
                    overlap_y = random.randint(
                        paste_y, paste_y + int(receipt.height * 0.4)
                    )  # Top 40% of receipt

                    # Ensure payment receipt stays within canvas bounds
                    overlap_x = max(
                        0, min(overlap_x, canvas_size[0] - payment_receipt.width)
                    )
                    overlap_y = max(
                        0, min(overlap_y, canvas_size[1] - payment_receipt.height)
                    )

                    # Add staple marks at the overlap point
                    staple_color = (120, 120, 120)  # Gray color for staple
                    staple_img = payment_receipt.copy()
                    staple_draw = ImageDraw.Draw(staple_img)

                    # Add 1-2 staple marks at corners or edges
                    num_staples = random.randint(1, 2)
                    for _ in range(num_staples):
                        # Choose staple position along edges
                        staple_x = random.randint(10, payment_receipt.width - 10)
                        staple_y = random.randint(10, payment_receipt.height - 10)

                        # 70% chance staple is at the top or bottom edge
                        if random.random() < 0.7:
                            if random.random() < 0.5:
                                staple_y = random.randint(5, 15)  # Top edge
                            else:
                                staple_y = random.randint(
                                    payment_receipt.height - 15,
                                    payment_receipt.height - 5,
                                )  # Bottom edge

                        # Draw a small circle to represent the staple
                        staple_size = 3
                        staple_draw.ellipse(
                            [
                                (staple_x - staple_size, staple_y - staple_size),
                                (staple_x + staple_size, staple_y + staple_size),
                            ],
                            fill=staple_color,
                        )

                    # Paste the payment receipt on top
                    canvas.paste(staple_img, (overlap_x, overlap_y))

                except Exception as e:
                    print(f"Error adding stapled payment receipt: {e}")

        except Exception as e:
            print(f"Error creating receipt {i}: {e}")
            continue

    return canvas, actual_count


def main():
    """
    Generate synthetic receipt collages for training vision transformer models.

    This script creates synthetic training data by generating fake receipts from scratch
    and placing them on a white background arranged in a grid pattern.

    Key features:
    - Completely synthetic receipts with realistic text content
    - Simple white background for all collages
    - Grid-based placement with minimal jitter
    - Minimal rotation for slight variation
    - No overlaps between receipts
    - Option to add stapled payment receipts on top of main receipts (--stapled_ratio)
    - 0-receipt examples use Australian tax documents in portrait orientation
    - Random portrait or landscape orientations to mimic real phone photos

    The class distribution can be specified via the --count_probs argument.
    Tax documents (for 0-receipt examples) are always in portrait orientation
    regardless of the collage orientation.
    """
    parser = argparse.ArgumentParser(description="Create synthetic receipt collages")
    parser.add_argument(
        "--output_dir",
        default="synthetic_receipts",
        help="Directory to save collage images",
    )
    parser.add_argument(
        "--num_collages", type=int, default=300, help="Number of collages to create"
    )
    parser.add_argument(
        "--canvas_width",
        type=int,
        default=1600,
        help="Width of the collage canvas (will be swapped for portrait orientation)",
    )
    parser.add_argument(
        "--canvas_height",
        type=int,
        default=1200,
        help="Height of the collage canvas (will be swapped for portrait orientation)",
    )
    parser.add_argument(
        "--count_probs",
        type=str,
        default="0.2,0.2,0.2,0.2,0.1,0.1",
        help="Comma-separated probabilities for 0,1,2,3,4,5 receipts",
    )
    parser.add_argument(
        "--stapled_ratio",
        type=float,
        default=0.0,
        help="Proportion of receipts with payment receipts stapled on top (0.0-1.0)",
    )

    args = parser.parse_args()

    # Convert path to Path object
    output_dir = Path(args.output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse probability distribution
    try:
        # Parse probabilities from command line
        count_probs = [float(p) for p in args.count_probs.split(",")]

        # Normalize to ensure they sum to 1
        prob_sum = sum(count_probs)
        if prob_sum <= 0:
            raise ValueError("Probabilities must sum to a positive value")
        count_probs = [p / prob_sum for p in count_probs]
        print(f"Using receipt count distribution: {count_probs}")

    except (ValueError, AttributeError) as e:
        print(f"Warning: Invalid probability format: {e}")
        # Default to uniform distribution
        count_probs = [1 / 6] * 6
        print(f"Using default uniform distribution: {count_probs}")

    # Count distribution for verification
    count_distribution = {i: 0 for i in range(len(count_probs))}

    # Always use white background
    print("Using white background for all collages (255, 255, 255)")

    # Print stapled receipt info
    if args.stapled_ratio > 0:
        print(
            f"Adding stapled payment receipts to {args.stapled_ratio * 100:.1f}% of receipts"
        )

    # Only create collages if num_collages > 0
    if args.num_collages > 0:
        # Create collages
        for i in tqdm(range(args.num_collages), desc="Creating synthetic collages"):
            # Randomly choose between portrait and landscape orientation
            is_portrait = random.choice([True, False])

            # If portrait, swap width and height
            if is_portrait:
                canvas_size = (
                    args.canvas_height,
                    args.canvas_width,
                )  # Portrait (height > width)
            else:
                canvas_size = (
                    args.canvas_width,
                    args.canvas_height,
                )  # Landscape (width > height)

            # Select receipt count based on probability distribution
            receipt_count = random.choices(
                list(range(len(count_probs))), weights=count_probs
            )[0]

            # Create collage with synthetic receipts
            collage, actual_count = create_receipt_collage(
                canvas_size=canvas_size,
                receipt_count=receipt_count,
                stapled_ratio=args.stapled_ratio,
            )

            # Track distribution
            count_distribution[actual_count] += 1

            # Save the collage using the original naming convention
            output_path = output_dir / f"synthetic_{i:03d}_{actual_count}_receipts.jpg"
            collage.save(output_path, "JPEG", quality=95)

        # Report final receipt count distribution
        print("\nActual receipt count distribution:")
        for count, freq in sorted(count_distribution.items()):
            percentage = freq / args.num_collages * 100
            print(f"  {count} receipts: {freq} collages ({percentage:.1f}%)")

        print(f"\nCreated {args.num_collages} synthetic collages in {output_dir}")

    # Generate 100 individual receipts for testing
    print("\nGenerating 100 individual receipt samples...")
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    for i in range(100):
        receipt = generate_synthetic_receipt()
        receipt_path = samples_dir / f"receipt_sample_{i + 1}.jpg"
        receipt.save(receipt_path, "JPEG", quality=95)

    print(f"Created 100 individual receipt samples in {samples_dir}")


if __name__ == "__main__":
    main()
