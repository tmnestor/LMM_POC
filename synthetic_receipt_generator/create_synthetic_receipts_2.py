#!/usr/bin/env python3
"""
Generate synthetic receipt images for training.

This script creates synthetic receipt images with varying formats, content, and appearances
to serve as training data for the receipt counter model.
"""

import argparse
import random
from pathlib import Path

import numpy as np
from data.data_generators.receipt_processor import create_blank_image
from PIL import Image, ImageDraw, ImageFont


def create_receipt_image(width=2000, height=3000, items_count=None, max_items=20):
    """
    Create a single synthetic receipt image.

    Args:
        width: Width of the receipt
        height: Height of the receipt
        items_count: Optional fixed number of items on receipt
        max_items: Maximum number of items if items_count is None

    Returns:
        PIL Image of the synthetic receipt
    """
    # Create receipt with original dimensions
    receipt = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(receipt)

    # Try to load fonts with increased sizes for higher resolution
    try:
        # Try to use system fonts at large sizes for high-quality receipts
        font_header = ImageFont.truetype("Arial Bold", 60)
        font_body = ImageFont.truetype("Arial", 40)
        font_small = ImageFont.truetype("Arial", 30)
    except IOError:
        # If specific fonts not available, try generic fonts
        try:
            font_header = ImageFont.truetype("DejaVuSans-Bold", 60)
            font_body = ImageFont.truetype("DejaVuSans", 40)
            font_small = ImageFont.truetype("DejaVuSans", 30)
        except IOError:
            # Fall back to default font if needed
            font_header = ImageFont.load_default()
            font_body = ImageFont.load_default()
            font_small = ImageFont.load_default()

    # Generate more realistic receipt content
    receipt_types = ["standard", "detailed", "minimal", "fancy"]
    receipt_type = random.choice(receipt_types)

    # Add store name and header
    store_names = [
        "GROCERY WORLD",
        "SUPERMARKET PLUS",
        "FOOD MART",
        "MARKET PLACE",
        "CONVENIENCE STORE",
        "FRESH FOODS",
        "MEGA MART",
        "VALUE STORE",
        "QUICK SHOP",
        "DAILY GOODS",
        "FAMILY MARKET",
    ]
    store_name = random.choice(store_names)

    # Create more realistic date formatting
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    month = random.choice(months)
    day = random.randint(1, 28)
    year = random.randint(2020, 2023)
    hour = random.randint(8, 21)
    minute = random.randint(0, 59)
    ampm = "AM" if hour < 12 else "PM"
    if hour > 12:
        hour -= 12
    if hour == 0:
        hour = 12

    date_formats = [
        f"{month} {day}, {year}",
        f"{month}/{day}/{year}",
        f"{day}/{month}/{year}",
        f"{day}-{month}-{year}",
        f"{month} {day} {year}",
        f"{day}.{month}.{year}",
    ]

    time_formats = [
        f"{hour}:{minute:02d} {ampm}",
        f"{hour}:{minute:02d}",
        f"{hour:02d}:{minute:02d}",
        f"{hour}:{minute:02d}{ampm.lower()}",
    ]

    date_str = random.choice(date_formats)
    time_str = random.choice(time_formats)

    # Generate receipt number/transaction ID
    receipt_id = f"#{random.randint(100000, 999999)}"

    # Different receipt layouts based on type
    # Fill in these variables in advance for use in multiple receipt types
    # Generate store address (used in multiple receipt types)
    street_num = random.randint(100, 9999)
    streets = [
        "Main St",
        "Oak Avenue",
        "Park Road",
        "Market Street",
        "Broadway",
        "First Avenue",
    ]
    street = random.choice(streets)
    cities = [
        "Sydney",
        "Melbourne",
        "Brisbane",
        "Perth",
        "Adelaide",
        "Canberra",
        "Hobart",
    ]
    city = random.choice(cities)
    states = ["NSW", "VIC", "QLD", "WA", "SA", "ACT", "TAS", "NT"]
    state = random.choice(states)
    post_code = random.randint(1000, 9999)
    store_address = f"{street_num} {street}, {city}, {state} {post_code}"

    # Phone number (Australian format)
    area_code = random.randint(2, 9)
    phone_prefix = random.randint(1000, 9999)
    phone_suffix = random.randint(1000, 9999)
    phone = f"(0{area_code}) {phone_prefix} {phone_suffix}"

    # Date and time more visible formats
    hour = random.randint(8, 21)
    minute = random.randint(0, 59)
    am_pm = "AM" if hour < 12 else "PM"
    time_str_display = f"{hour}:{minute:02d} {am_pm}"

    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(2020, 2023)
    date_str_display = f"{day}/{month}/{year}"

    # Payment methods - select one for this receipt
    payment_methods = ["VISA", "MASTERCARD", "AMEX", "CASH", "DEBIT"]
    payment = random.choice(payment_methods)

    # Card number (partially masked) if it's a card payment
    if payment != "CASH":
        card_num = "XXXX-XXXX-XXXX-" + str(random.randint(1000, 9999))
    else:
        card_num = ""

    # Auth code for card transactions
    auth_code = f"{random.randint(100000, 999999)}"

    if receipt_type == "standard":
        # Standard receipt layout
        draw.text((width // 2 - 200, 80), store_name, fill="black", font=font_header)
        draw.text((width // 2 - 120, 160), "RECEIPT", fill="black", font=font_header)

        # Use the pre-generated store address

        # Add address centered
        address_x = max(50, width // 2 - len(store_address) * 8)
        draw.text((address_x, 200), store_address, fill="black", font=font_small)

        # Draw divider
        draw.line([(100, 240), (width - 100, 240)], fill="black", width=3)

        # Add date and time
        draw.text((150, 280), f"DATE: {date_str_display}", fill="black", font=font_body)
        draw.text((150, 340), f"TIME: {time_str_display}", fill="black", font=font_body)
        draw.text(
            (width - 400, 280), f"TRANS ID: {receipt_id}", fill="black", font=font_body
        )

        # Payment method using the pre-generated value
        draw.text(
            (width - 400, 340), f"PAYMENT: {payment}", fill="black", font=font_body
        )

        # Draw second divider
        draw.line([(100, 400), (width - 100, 400)], fill="black", width=3)

        # Set starting position for items
        y_pos = 460

        # Column headers
        draw.text((150, y_pos), "ITEM", fill="black", font=font_body)
        draw.text((width - 500, y_pos), "QTY", fill="black", font=font_body)
        draw.text((width - 350, y_pos), "PRICE", fill="black", font=font_body)
        draw.text((width - 200, y_pos), "TOTAL", fill="black", font=font_body)

        y_pos += 60
        # Draw header divider
        draw.line([(100, y_pos - 20), (width - 100, y_pos - 20)], fill="black", width=2)

    elif receipt_type == "detailed":
        # More detailed receipt with company info
        draw.text((width // 2 - 200, 80), store_name, fill="black", font=font_header)
        draw.text(
            (width // 2 - 250, 160), "PAYMENT RECEIPT", fill="black", font=font_header
        )

        # Add random street address
        streets = [
            "Main St",
            "Oak Avenue",
            "Park Road",
            "Market Street",
            "Broadway",
            "First Avenue",
        ]
        street_num = random.randint(100, 9999)
        street = random.choice(streets)
        cities = [
            "Sydney",
            "Melbourne",
            "Brisbane",
            "Perth",
            "Adelaide",
            "Canberra",
            "Hobart",
        ]
        city = random.choice(cities)
        states = ["NSW", "VIC", "QLD", "WA", "SA", "ACT", "TAS", "NT"]
        state = random.choice(states)
        post_code = random.randint(1000, 9999)

        address = f"{street_num} {street}, {city}, {state} {post_code}"
        address_x = max(50, width // 2 - len(address) * 10)
        draw.text((address_x, 220), address, fill="black", font=font_small)

        # Add phone number
        area_code = random.randint(2, 9)
        phone_prefix = random.randint(1000, 9999)
        phone_suffix = random.randint(1000, 9999)
        phone = f"(0{area_code}) {phone_prefix} {phone_suffix}"
        phone_x = max(50, width // 2 - len(phone) * 10)
        draw.text((phone_x, 260), phone, fill="black", font=font_small)

        # Divider
        draw.line([(100, 320), (width - 100, 320)], fill="black", width=3)

        # Add date and transaction details in left column
        draw.text((150, 360), "DATE:", fill="black", font=font_body)
        draw.text((400, 360), f"{date_str}", fill="black", font=font_body)

        draw.text((150, 420), "TIME:", fill="black", font=font_body)
        draw.text((400, 420), f"{time_str}", fill="black", font=font_body)

        draw.text((150, 480), "TRANS ID:", fill="black", font=font_body)
        draw.text((400, 480), f"{receipt_id}", fill="black", font=font_body)

        # Add payment method details
        payment_methods = ["VISA", "MASTERCARD", "AMEX", "CASH", "DEBIT"]
        payment = random.choice(payment_methods)
        draw.text((150, 540), "METHOD:", fill="black", font=font_body)
        draw.text((400, 540), f"{payment}", fill="black", font=font_body)

        # Add card number (partially masked) if it's a card
        if payment != "CASH":
            card_number = "XXXX-XXXX-XXXX-" + str(random.randint(1000, 9999))
            auth_code = f"AUTH: {random.randint(100000, 999999)}"

            # Draw card details
            draw.text((150, 600), "CARD:", fill="black", font=font_body)
            draw.text((400, 600), f"{card_number}", fill="black", font=font_body)

            draw.text((150, 660), "AUTH:", fill="black", font=font_body)
            draw.text((400, 660), f"{auth_code}", fill="black", font=font_body)

            # Add "APPROVED" text
            draw.text((400, 720), "APPROVED", fill="black", font=font_body)
            draw.text((400, 780), "THANK YOU", fill="black", font=font_body)

        # Draw a second divider
        draw.line([(100, 840), (width - 100, 840)], fill="black", width=3)

        # Start the item list
        y_pos = 900

        # Column headers aligned to the right side
        draw.text((width - 700, y_pos), "ITEM", fill="black", font=font_body)
        draw.text((width - 500, y_pos), "QTY", fill="black", font=font_body)
        draw.text((width - 350, y_pos), "PRICE", fill="black", font=font_body)
        draw.text((width - 200, y_pos), "TOTAL", fill="black", font=font_body)

        y_pos += 60
        # Draw header divider
        draw.line(
            [(width // 2, y_pos - 20), (width - 100, y_pos - 20)], fill="black", width=2
        )

    elif receipt_type == "minimal":
        # Minimalist receipt
        draw.text((width // 2 - 150, 80), store_name, fill="black", font=font_header)

        # Simple divider
        draw.line([(width // 4, 160), (width * 3 // 4, 160)], fill="black", width=2)

        # Basic transaction info
        y_pos = 200
        draw.text((150, y_pos), f"Date: {date_str}", fill="black", font=font_body)
        draw.text((150, y_pos + 60), f"Time: {time_str}", fill="black", font=font_body)
        draw.text((150, y_pos + 120), f"ID: {receipt_id}", fill="black", font=font_body)

        # Payment method
        payment_methods = ["VISA", "MASTERCARD", "AMEX", "CASH", "DEBIT"]
        payment = random.choice(payment_methods)
        draw.text(
            (150, y_pos + 180), f"Method: {payment}", fill="black", font=font_body
        )

        # Start items further down
        y_pos = 460

    else:  # fancy
        # More stylized receipt
        # Add a decorative header
        for i in range(5):
            y = 40 + i * 20
            draw.line([(100, y), (width - 100, y)], fill="black", width=1)

        # Store name in large font
        draw.text((width // 2 - 220, 150), store_name, fill="black", font=font_header)

        # Draw a decorative separator
        separator_y = 240
        for i in range(10):
            x1 = width // 2 - 300 + i * 60
            x2 = x1 + 30
            draw.line([(x1, separator_y), (x2, separator_y)], fill="black", width=2)

        # Add receipt title
        draw.text(
            (width // 2 - 160, 280), "PURCHASE RECEIPT", fill="black", font=font_header
        )

        # Draw decorative box around date/time
        box_top = 360
        box_bottom = 520
        draw.rectangle(
            [(150, box_top), (width - 150, box_bottom)], outline="black", width=2
        )

        # Add date and time in the box with conspicuous display values
        draw.text((200, box_top + 30), "DATE:", fill="black", font=font_body)
        draw.text(
            (500, box_top + 30), f"{date_str_display}", fill="black", font=font_body
        )

        draw.text((200, box_top + 90), "TIME:", fill="black", font=font_body)
        draw.text(
            (500, box_top + 90), f"{time_str_display}", fill="black", font=font_body
        )

        # Add transaction ID
        draw.text(
            (width - 600, box_top + 30), "RECEIPT #:", fill="black", font=font_body
        )
        draw.text(
            (width - 350, box_top + 30), f"{receipt_id}", fill="black", font=font_body
        )

        # Payment method - use pre-defined value
        draw.text((width - 600, box_top + 90), "PAYMENT:", fill="black", font=font_body)
        draw.text(
            (width - 350, box_top + 90), f"{payment}", fill="black", font=font_body
        )

        # Store address (add it to the fancy layout too)
        draw.text(
            (width // 2 - len(store_address) * 7, 200),
            store_address,
            fill="black",
            font=font_small,
        )

        # Add a few more transaction details
        if payment != "CASH":
            # Card number for card payments
            card_number = "XXXX-XXXX-XXXX-" + str(random.randint(1000, 9999))
            draw.text(
                (width - 600, box_top + 150), "CARD:", fill="black", font=font_body
            )
            draw.text(
                (width - 350, box_top + 150),
                f"{card_number}",
                fill="black",
                font=font_body,
            )

        # Start items list further down
        y_pos = 580

        # Add stylish column headers with background
        header_bg = (240, 240, 240)  # Light gray
        draw.rectangle([(100, y_pos - 10), (width - 100, y_pos + 50)], fill=header_bg)

        draw.text((150, y_pos), "PRODUCT", fill="black", font=font_body)
        draw.text((width - 500, y_pos), "QTY", fill="black", font=font_body)
        draw.text((width - 350, y_pos), "PRICE", fill="black", font=font_body)
        draw.text((width - 200, y_pos), "TOTAL", fill="black", font=font_body)

        y_pos += 80

    # Create expanded list of realistic items with prices
    items = [
        ("Milk", 3.99, "1 gal"),
        ("Bread", 2.49, "1 loaf"),
        ("Eggs", 3.29, "dozen"),
        ("Cheese", 4.99, "8 oz"),
        ("Apples", 1.49, "lb"),
        ("Bananas", 0.59, "lb"),
        ("Chicken", 6.99, "lb"),
        ("Ground Beef", 5.99, "lb"),
        ("Rice", 3.49, "2 lb"),
        ("Pasta", 1.99, "16 oz"),
        ("Cereal", 4.29, "18 oz"),
        ("Coffee", 7.99, "12 oz"),
        ("Tea", 3.99, "20 ct"),
        ("Chocolate", 2.99, "3.5 oz"),
        ("Yogurt", 1.29, "6 oz"),
        ("Juice", 3.89, "64 oz"),
        ("Soda", 1.99, "2 liter"),
        ("Water", 4.99, "24 pk"),
        ("Potato Chips", 3.49, "8 oz"),
        ("Ice Cream", 4.99, "1 qt"),
        ("Frozen Pizza", 5.99, "each"),
        ("Toilet Paper", 6.99, "12 roll"),
        ("Paper Towels", 4.99, "6 roll"),
        ("Dish Soap", 2.79, "16 oz"),
        ("Laundry Detergent", 9.99, "50 oz"),
        ("Shampoo", 4.99, "12 oz"),
        ("Toothpaste", 3.49, "6 oz"),
        ("Batteries", 8.99, "8 pack"),
        ("Light Bulbs", 7.99, "4 pack"),
    ]

    # Determine number of items
    if items_count is None:
        items_count = random.randint(8, max_items)

    subtotal = 0
    item_entries = []

    # Select random items and quantities
    selected_items = random.sample(items, min(items_count, len(items)))
    if len(selected_items) < items_count:
        # Add some duplicates if we need more items
        additional = random.choices(items, k=items_count - len(selected_items))
        selected_items.extend(additional)

    # Calculate item prices and totals
    for item_name, base_price, unit in selected_items:
        # Randomize quantity (1-5)
        qty = random.randint(1, 3)
        # Small random variation in price
        price_variation = random.uniform(0.95, 1.05)
        price = round(base_price * price_variation, 2)
        total = round(price * qty, 2)
        subtotal += total

        # Store the item info
        item_entries.append((item_name, qty, price, total, unit))

    # Draw the items
    for item_name, qty, price, total, unit in item_entries:
        if y_pos > height - 400:  # Ensure we have space for footer
            break

        if receipt_type == "standard":
            draw.text((150, y_pos), f"{item_name}", fill="black", font=font_body)
            draw.text((width - 500, y_pos), f"{qty}", fill="black", font=font_body)
            draw.text(
                (width - 350, y_pos), f"${price:.2f}", fill="black", font=font_body
            )
            draw.text(
                (width - 200, y_pos), f"${total:.2f}", fill="black", font=font_body
            )

        elif receipt_type == "detailed":
            draw.text(
                (width - 700, y_pos),
                f"{item_name} ({unit})",
                fill="black",
                font=font_body,
            )
            draw.text((width - 500, y_pos), f"{qty}", fill="black", font=font_body)
            draw.text(
                (width - 350, y_pos), f"${price:.2f}", fill="black", font=font_body
            )
            draw.text(
                (width - 200, y_pos), f"${total:.2f}", fill="black", font=font_body
            )

        elif receipt_type == "minimal":
            draw.text((150, y_pos), f"{item_name}", fill="black", font=font_body)
            draw.text(
                (width - 200, y_pos), f"${total:.2f}", fill="black", font=font_body
            )

        else:  # fancy
            # Alternating row colors
            if (y_pos // 60) % 2 == 0:
                row_bg = (248, 248, 248)  # Very light gray
                draw.rectangle(
                    [(100, y_pos - 10), (width - 100, y_pos + 50)], fill=row_bg
                )

            draw.text(
                (150, y_pos), f"{item_name} ({unit})", fill="black", font=font_body
            )
            draw.text((width - 500, y_pos), f"{qty}", fill="black", font=font_body)
            draw.text(
                (width - 350, y_pos), f"${price:.2f}", fill="black", font=font_body
            )
            draw.text(
                (width - 200, y_pos), f"${total:.2f}", fill="black", font=font_body
            )

        y_pos += 60

    # Add separator line before totals
    draw.line([(100, y_pos + 10), (width - 100, y_pos + 10)], fill="black", width=2)
    y_pos += 30

    # Calculate tax and total
    tax_rate = random.uniform(0.05, 0.095)  # 5-9.5% tax
    tax = round(subtotal * tax_rate, 2)
    total = subtotal + tax

    # Format tax rate as percentage
    tax_percentage = f"({tax_rate * 100:.1f}%)"

    # Add totals section
    if receipt_type in ["standard", "minimal"]:
        y_pos += 20
        draw.text((width - 400, y_pos), "Subtotal:", fill="black", font=font_body)
        draw.text(
            (width - 200, y_pos), f"${subtotal:.2f}", fill="black", font=font_body
        )

        y_pos += 60
        draw.text(
            (width - 400, y_pos), f"Tax {tax_percentage}:", fill="black", font=font_body
        )
        draw.text((width - 200, y_pos), f"${tax:.2f}", fill="black", font=font_body)

        y_pos += 60
        draw.text((width - 400, y_pos), "TOTAL:", fill="black", font=font_header)
        draw.text((width - 200, y_pos), f"${total:.2f}", fill="black", font=font_header)

    elif receipt_type == "detailed":
        # Calculation box
        box_left = width - 600
        box_top = y_pos
        box_width = 500
        box_height = 240
        draw.rectangle(
            [(box_left, box_top), (box_left + box_width, box_top + box_height)],
            outline="black",
            width=2,
        )

        # Add totals inside box
        y_pos += 30
        draw.text((box_left + 50, y_pos), "Subtotal:", fill="black", font=font_body)
        draw.text(
            (box_left + 300, y_pos), f"${subtotal:.2f}", fill="black", font=font_body
        )

        y_pos += 60
        draw.text(
            (box_left + 50, y_pos),
            f"Tax {tax_percentage}:",
            fill="black",
            font=font_body,
        )
        draw.text((box_left + 300, y_pos), f"${tax:.2f}", fill="black", font=font_body)

        # Line inside box
        draw.line(
            [(box_left + 50, y_pos + 40), (box_left + box_width - 50, y_pos + 40)],
            fill="black",
            width=1,
        )

        y_pos += 90
        draw.text((box_left + 50, y_pos), "TOTAL:", fill="black", font=font_header)
        draw.text(
            (box_left + 300, y_pos), f"${total:.2f}", fill="black", font=font_header
        )

        y_pos = box_top + box_height + 40

    else:  # fancy
        # Bordered total section
        y_pos += 20
        box_width = 600
        box_left = width - 100 - box_width

        # Subtotal box
        draw.rectangle(
            [(box_left, y_pos), (width - 100, y_pos + 60)], outline="black", width=1
        )
        draw.text(
            (box_left + 50, y_pos + 10), "Subtotal:", fill="black", font=font_body
        )
        draw.text(
            (width - 200, y_pos + 10), f"${subtotal:.2f}", fill="black", font=font_body
        )

        # Tax box
        y_pos += 60
        draw.rectangle(
            [(box_left, y_pos), (width - 100, y_pos + 60)], outline="black", width=1
        )
        draw.text(
            (box_left + 50, y_pos + 10),
            f"Tax {tax_percentage}:",
            fill="black",
            font=font_body,
        )
        draw.text(
            (width - 200, y_pos + 10), f"${tax:.2f}", fill="black", font=font_body
        )

        # Total box (highlighted)
        y_pos += 60
        total_bg = (240, 240, 240)  # Light gray
        draw.rectangle(
            [(box_left, y_pos), (width - 100, y_pos + 70)],
            fill=total_bg,
            outline="black",
            width=2,
        )
        draw.text((box_left + 50, y_pos + 15), "TOTAL:", fill="black", font=font_header)
        draw.text(
            (width - 220, y_pos + 15), f"${total:.2f}", fill="black", font=font_header
        )

        y_pos += 90

    # Add footer
    y_pos += 80

    # Different footers based on receipt type
    if receipt_type == "standard":
        draw.text(
            (width // 2 - 200, y_pos),
            "THANK YOU FOR SHOPPING WITH US",
            fill="black",
            font=font_body,
        )
        y_pos += 60
        draw.text(
            (width // 2 - 100, y_pos), "PLEASE COME AGAIN", fill="black", font=font_body
        )

    elif receipt_type == "detailed":
        # Draw box for footer
        box_top = y_pos
        box_height = 180
        draw.rectangle(
            [(150, box_top), (width - 150, box_top + box_height)],
            outline="black",
            width=1,
        )

        draw.text(
            (width // 2 - 250, y_pos + 30),
            "THANK YOU FOR YOUR BUSINESS",
            fill="black",
            font=font_body,
        )
        y_pos += 80
        draw.text(
            (width // 2 - 180, y_pos),
            "VISIT US ONLINE AT:",
            fill="black",
            font=font_body,
        )
        y_pos += 60

        # Generate fake website
        website = f"www.{store_name.lower().replace(' ', '')}.com"
        website = "".join(c for c in website if c.isalnum() or c in [".", "-", "/"])
        draw.text(
            (width // 2 - len(website) * 10, y_pos),
            website,
            fill="black",
            font=font_body,
        )

    elif receipt_type == "minimal":
        draw.text((width // 2 - 140, y_pos), "THANK YOU", fill="black", font=font_body)

    else:  # fancy
        # Draw decorative line
        for i in range(10):
            x1 = width // 2 - 300 + i * 60
            x2 = x1 + 30
            draw.line([(x1, y_pos), (x2, y_pos)], fill="black", width=2)

        y_pos += 40
        draw.text(
            (width // 2 - 300, y_pos),
            "THANK YOU FOR YOUR PURCHASE",
            fill="black",
            font=font_header,
        )
        y_pos += 60

        # Add a random message
        messages = [
            "We appreciate your business!",
            "Please come back soon!",
            "Save your receipt for returns",
            "Rate our service online",
            "Join our loyalty program!",
            "Follow us on social media",
        ]
        message = random.choice(messages)
        draw.text(
            (width // 2 - len(message) * 8, y_pos),
            message,
            fill="black",
            font=font_body,
        )

        # Add decorative bottom border
        y_pos += 80
        for i in range(5):
            y = y_pos + i * 20
            draw.line([(100, y), (width - 100, y)], fill="black", width=1)

    # Add minor rotation for realism
    rotation = random.uniform(-1, 1)
    receipt = receipt.rotate(rotation, expand=True, fillcolor="white")

    return receipt


def create_tax_document(image_size=2048):
    """
    Creates an Australian Taxation Office document (not a receipt).

    Args:
        image_size: Size of the output image

    Returns:
        PIL Image containing a tax document
    """
    # Create a white document
    doc = Image.new("RGB", (image_size, image_size), color=(255, 255, 255))
    draw = ImageDraw.Draw(doc)

    # Add ATO-style header
    # Draw blue ATO banner at top
    ato_blue = (0, 51, 160)  # ATO blue color
    draw.rectangle([(0, 0), (image_size, 60)], fill=ato_blue)

    # Load font or use default
    try:
        header_font = ImageFont.truetype("Arial Bold", 60)
    except OSError:
        header_font = ImageFont.load_default()

    try:
        body_font = ImageFont.truetype("Arial", 40)
    except OSError:
        body_font = ImageFont.load_default()

    try:
        small_font = ImageFont.truetype("Arial", 30)
    except OSError:
        small_font = ImageFont.load_default()

    # Draw ATO text
    draw.text((20, 20), "Australian Taxation Office", fill="white", font=header_font)

    # Choose document type
    doc_types = [
        "Medicare Levy Exemption Certificate",
        "Business Activity Statement",
        "Income Tax Assessment Notice",
        "PAYG Payment Summary",
        "Tax Return Summary",
        "Superannuation Statement",
        "Notice of Assessment",
    ]
    doc_type = random.choice(doc_types)

    # Draw document title and period
    year = random.randint(2020, 2023)
    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    month = random.choice(month_names)

    # Center the title
    draw.text((image_size // 2 - 400, 140), doc_type, fill="black", font=header_font)

    # Choose period format based on document type
    if "Medicare" in doc_type or "Superannuation" in doc_type:
        period_text = f"For the financial year ending 30 June {year}"
    elif "Business Activity" in doc_type:
        quarter = random.choice(
            [
                "January - March",
                "April - June",
                "July - September",
                "October - December",
            ]
        )
        period_text = f"For the period: {quarter} {year}"
    else:
        period_text = f"For the period: {month} {year}"

    draw.text((image_size // 2 - 300, 220), period_text, fill="black", font=body_font)

    # Draw taxpayer details section
    draw.rectangle(
        [(image_size // 5, 320), (image_size * 4 // 5, 480)], outline="black", width=2
    )
    draw.text(
        (image_size // 5 + 30, 340), "Taxpayer Details", fill="black", font=body_font
    )

    # Generate random name with some characters replaced with X
    first_names = [
        "John",
        "Sarah",
        "Michael",
        "Emma",
        "Robert",
        "Jennifer",
        "David",
        "Emily",
    ]
    last_names = [
        "Smith",
        "Johnson",
        "Williams",
        "Brown",
        "Jones",
        "Miller",
        "Davis",
        "Wilson",
    ]
    name = f"{random.choice(first_names)} {random.choice(last_names)}"

    # Anonymize name by replacing some characters with X
    name_chars = list(name)
    replace_pos = random.sample(range(len(name)), random.randint(3, 5))
    for pos in replace_pos:
        if name_chars[pos] != " ":  # Don't replace spaces
            name_chars[pos] = "X"
    anonymized_name = "".join(name_chars)

    # Generate TFN with X
    tfn = f"{random.randint(100, 999)} {random.randint(100, 999)} {random.randint(100, 999)}"
    tfn_chars = list(tfn)
    replace_pos = random.sample(range(len(tfn)), random.randint(4, 6))
    for pos in replace_pos:
        if tfn_chars[pos] != " ":  # Don't replace spaces
            tfn_chars[pos] = "X"
    anonymized_tfn = "".join(tfn_chars)

    draw.text(
        (image_size // 5 + 30, 400),
        f"Name: {anonymized_name}",
        fill="black",
        font=body_font,
    )
    draw.text(
        (image_size // 5 + 30, 460),
        f"TFN: {anonymized_tfn}",
        fill="black",
        font=body_font,
    )

    # Draw content section based on document type
    content_top = 520
    content_height = 400
    draw.rectangle(
        [
            (image_size // 5, content_top),
            (image_size * 4 // 5, content_top + content_height),
        ],
        outline="black",
        width=2,
    )

    # Add content based on document type
    if "Medicare" in doc_type:
        draw.text(
            (image_size // 5 + 30, content_top + 30),
            "Exemption Details",
            fill="black",
            font=body_font,
        )
        draw.text(
            (image_size // 5 + 30, content_top + 90),
            f"Exemption Type:    Full Medicare Levy Exemption - Category {random.randint(1, 3)}",
            fill="black",
            font=small_font,
        )
        draw.text(
            (image_size // 5 + 30, content_top + 140),
            f"Certificate Number: M{random.randint(100000, 999999)}",
            fill="black",
            font=small_font,
        )
        draw.text(
            (image_size // 5 + 30, content_top + 190),
            f"Valid Period:       01/07/{year - 1} to 30/06/{year}",
            fill="black",
            font=small_font,
        )
        draw.text(
            (image_size // 5 + 30, content_top + 250),
            "This certificate confirms that the taxpayer named above is:",
            fill="black",
            font=small_font,
        )
        draw.text(
            (image_size // 5 + 60, content_top + 310),
            "A member of a diplomatic mission or consular post in Australia",
            fill="black",
            font=small_font,
        )

    elif "Business Activity" in doc_type:
        draw.text(
            (image_size // 5 + 30, content_top + 30),
            "GST and PAYG Summary",
            fill="black",
            font=body_font,
        )
        sales = random.randint(100000, 999999)
        gst_sales = int(sales * 0.1)
        purchases = random.randint(50000, sales - 10000)
        gst_purchases = int(purchases * 0.1)
        net_gst = gst_sales - gst_purchases
        payg = random.randint(10000, 50000)
        total = net_gst + payg

        draw.text(
            (image_size // 5 + 30, content_top + 90),
            "G1. Total sales (including GST):",
            fill="black",
            font=small_font,
        )
        draw.text(
            (image_size * 4 // 5 - 200, content_top + 90),
            f"${sales:,}",
            fill="black",
            font=small_font,
        )

        draw.text(
            (image_size // 5 + 30, content_top + 140),
            "G3. GST on sales:",
            fill="black",
            font=small_font,
        )
        draw.text(
            (image_size * 4 // 5 - 200, content_top + 140),
            f"${gst_sales:,}.00",
            fill="black",
            font=small_font,
        )

        draw.text(
            (image_size // 5 + 30, content_top + 190),
            "G10. Purchases (including GST):",
            fill="black",
            font=small_font,
        )
        draw.text(
            (image_size * 4 // 5 - 200, content_top + 190),
            f"${purchases:,}",
            fill="black",
            font=small_font,
        )

        draw.text(
            (image_size // 5 + 30, content_top + 240),
            "G11. GST on purchases:",
            fill="black",
            font=small_font,
        )
        draw.text(
            (image_size * 4 // 5 - 200, content_top + 240),
            f"${gst_purchases:,}.00",
            fill="black",
            font=small_font,
        )

        draw.line(
            [
                (image_size // 5 + 30, content_top + 290),
                (image_size * 4 // 5 - 30, content_top + 290),
            ],
            fill="black",
            width=1,
        )

        draw.text(
            (image_size // 5 + 30, content_top + 330),
            "Total amount payable:",
            fill="black",
            font=body_font,
        )
        draw.text(
            (image_size * 4 // 5 - 200, content_top + 330),
            f"${total:,}.00",
            fill=(200, 0, 0),
            font=body_font,
        )

    else:
        # Generic financial data for other document types
        draw.text(
            (image_size // 5 + 30, content_top + 30),
            f"{doc_type} Details",
            fill="black",
            font=body_font,
        )

        items = [
            ("Gross Income", random.randint(50000, 120000)),
            ("Deductions", random.randint(5000, 20000)),
            ("Taxable Income", 0),  # Will calculate below
            ("Tax Withheld", random.randint(10000, 30000)),
            ("Medicare Levy", random.randint(1000, 3000)),
            ("Tax Offset", random.randint(500, 2000)),
        ]

        # Calculate taxable income
        items[2] = ("Taxable Income", items[0][1] - items[1][1])

        y_pos = content_top + 90
        for label, amount in items:
            draw.text(
                (image_size // 5 + 30, y_pos),
                f"{label}:",
                fill="black",
                font=small_font,
            )
            draw.text(
                (image_size * 4 // 5 - 200, y_pos),
                f"${amount:,}",
                fill="black",
                font=small_font,
            )
            y_pos += 50

    # Add authorization footer
    if random.random() > 0.5:
        auth_text = "Authorized by: Services Australia"
    else:
        auth_text = "Authorized by: Australian Taxation Office"

    draw.text(
        (image_size // 5 + 30, content_top + content_height - 50),
        auth_text,
        fill="black",
        font=small_font,
    )

    # Add footer at the bottom of the document
    draw.text(
        (image_size // 2 - 300, image_size - 100),
        "This document must be retained for taxation purposes.",
        fill="black",
        font=small_font,
    )

    # Minor rotation to maintain document sharpness
    rotation = random.uniform(-0.5, 0.5)
    doc = doc.rotate(rotation, expand=True, fillcolor="white")

    # Resize while maintaining the required dimensions and quality
    doc = doc.resize((image_size, image_size), Image.LANCZOS)

    return doc


def create_receipt_collage(receipt_count, image_size=2048, stapled=False):
    """
    Create a collage with a specified number of receipts.

    Args:
        receipt_count: Number of receipts to include (0-5)
        image_size: Size of the output collage image
        stapled: Whether to create a stapled stack of receipts

    Returns:
        PIL Image containing the receipt collage
    """
    # Generate date and time displays for collages
    day = random.randint(1, 28)
    month = random.randint(1, 12)
    year = random.randint(2020, 2023)
    date_str_display = f"{day}/{month}/{year}"

    hour = random.randint(8, 21)
    minute = random.randint(0, 59)
    am_pm = "AM" if hour < 12 else "PM"
    time_str_display = f"{hour}:{minute:02d} {am_pm}"
    # Create blank image for collage
    collage = create_blank_image(image_size, image_size, "white")

    # If no receipts, return an ATO tax document (not a receipt)
    if receipt_count == 0:
        return create_tax_document(image_size)

    # Generate the requested number of receipts
    receipts = []
    for _ in range(receipt_count):
        # Create a receipt with random dimensions
        width = random.randint(300, 600)
        height = random.randint(800, 1500)
        items_count = random.randint(5, 15)
        receipt = create_receipt_image(width, height, items_count)

        # Resize receipt to fit in collage, maintaining aspect ratio
        scale_factor = min(
            image_size / 2 / receipt.width, image_size / 2 / receipt.height
        )
        new_width = int(receipt.width * scale_factor)
        new_height = int(receipt.height * scale_factor)
        receipt = receipt.resize((new_width, new_height), Image.LANCZOS)

        receipts.append(receipt)

    # For stapled receipts, create a stack with slight offsets
    if stapled and receipt_count > 1:
        # Find the largest receipt to determine overall dimensions
        max_width = max(r.width for r in receipts)
        max_height = max(r.height for r in receipts)

        # Center position for the stack
        center_x = (image_size - max_width) // 2
        center_y = (image_size - max_height) // 2

        # Place receipts with small offsets
        for _idx, receipt in enumerate(receipts):
            # Calculate offset for this receipt in the stack
            offset_x = random.randint(-10, 10)
            offset_y = random.randint(-10, 10)

            # Ensure the receipt stays within bounds
            x_pos = max(10, min(center_x + offset_x, image_size - receipt.width - 10))
            y_pos = max(10, min(center_y + offset_y, image_size - receipt.height - 10))

            # Paste the receipt onto the collage
            collage.paste(receipt, (x_pos, y_pos))

        # Add a "staple" mark at the top or side
        draw = ImageDraw.Draw(collage)
        if random.random() > 0.5:  # Top staple
            staple_x = center_x + max_width // 2
            staple_y = center_y - 5
            draw.line(
                [(staple_x - 5, staple_y), (staple_x + 5, staple_y)],
                fill="black",
                width=2,
            )
            draw.line(
                [(staple_x - 3, staple_y - 3), (staple_x - 3, staple_y + 3)],
                fill="black",
                width=2,
            )
            draw.line(
                [(staple_x + 3, staple_y - 3), (staple_x + 3, staple_y + 3)],
                fill="black",
                width=2,
            )
        else:  # Side staple
            staple_x = center_x - 5
            staple_y = center_y + max_height // 2
            draw.line(
                [(staple_x, staple_y - 5), (staple_x, staple_y + 5)],
                fill="black",
                width=2,
            )
            draw.line(
                [(staple_x - 3, staple_y - 3), (staple_x + 3, staple_y - 3)],
                fill="black",
                width=2,
            )
            draw.line(
                [(staple_x - 3, staple_y + 3), (staple_x + 3, staple_y + 3)],
                fill="black",
                width=2,
            )
    else:
        # Place receipts in collage (distributed across the image)
        for idx, receipt in enumerate(receipts):
            if receipt_count == 1:
                # Center the receipt
                x_pos = (image_size - receipt.width) // 2
                y_pos = (image_size - receipt.height) // 2

                # Make sure the receipt has complete data by drawing directly on it
                # Create a new image where we can place the receipt with all values filled
                receipt_with_values = create_blank_image(
                    image_size, image_size, "white"
                )
                draw = ImageDraw.Draw(receipt_with_values)

                # Draw the receipt first
                receipt_with_values.paste(receipt, (x_pos, y_pos))

                # Get center coordinates and add missing information
                center_x = image_size // 2
                y_start = y_pos + receipt.height // 4

                # Try to use nice fonts
                try:
                    font = ImageFont.truetype("Arial", 40)
                    small_font = ImageFont.truetype("Arial", 30)
                except OSError:
                    font = ImageFont.load_default()
                    small_font = font

                # Add store info above the receipt
                draw.text(
                    (center_x - 150, y_pos - 100),
                    f"DATE: {date_str_display}",
                    fill="black",
                    font=font,
                )
                draw.text(
                    (center_x - 150, y_pos - 50),
                    f"TIME: {time_str_display}",
                    fill="black",
                    font=font,
                )

                # Return this composite image instead of the original receipt
                collage = receipt_with_values
                continue  # Skip the normal pasting code below
            else:
                # Distribute receipts across the image
                if idx % 2 == 0:  # Left side
                    x_pos = random.randint(10, image_size // 2 - receipt.width - 10)
                else:  # Right side
                    x_pos = random.randint(
                        image_size // 2 + 10, image_size - receipt.width - 10
                    )

                y_pos = random.randint(10, image_size - receipt.height - 10)

            # Paste the receipt onto the collage
            collage.paste(receipt, (x_pos, y_pos))

    # No blur to maintain high image quality

    return collage


def generate_dataset(
    output_dir,
    num_collages=300,
    count_probs=None,
    image_size=2048,
    stapled_ratio=0.0,
    seed=42,
):
    """
    Generate a dataset of receipt collages with varying receipt counts.

    Args:
        output_dir: Directory to save the generated images
        num_collages: Number of collage images to generate
        count_probs: Probability distribution for number of receipts (0-5)
        image_size: Size of output images
        stapled_ratio: Ratio of images that should have stapled receipts (0.0-1.0)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with image filenames and receipt counts
    """
    import pandas as pd

    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Create output directories
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Default distribution if not provided
    if count_probs is None:
        count_probs = [
            0.3,
            0.3,
            0.2,
            0.1,
            0.1,
            0.0,
        ]  # Probabilities for 0, 1, 2, 3, 4, 5 receipts

    # Normalize probabilities to sum to 1
    count_probs = np.array(count_probs)
    count_probs = count_probs / count_probs.sum()

    # Generate collages
    data = []

    for i in range(num_collages):
        # Determine number of receipts based on probability distribution
        receipt_count = np.random.choice(len(count_probs), p=count_probs)

        # Determine if this should be a stapled collage
        # Only makes sense for multiple receipts
        is_stapled = False
        if receipt_count > 1 and random.random() < stapled_ratio:
            is_stapled = True

        # Create collage
        collage = create_receipt_collage(receipt_count, image_size, stapled=is_stapled)

        # Save image
        filename = f"receipt_collage_{i:05d}.png"
        collage.save(images_dir / filename)

        # Add to dataset
        data.append(
            {
                "filename": filename,
                "receipt_count": receipt_count,
                "is_stapled": is_stapled,
            }
        )

        # Progress update
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_collages} collages")

    # Create and save metadata
    df = pd.DataFrame(data)
    df.to_csv(output_dir / "metadata.csv", index=False)

    # Print some stats
    print(f"Dataset generation complete: {num_collages} images")
    print(
        f"Distribution of receipt counts: {df['receipt_count'].value_counts().sort_index()}"
    )
    print(f"ATO tax documents for 0 receipts: {len(df[df['receipt_count'] == 0])}")
    print(
        f"High-resolution images: {image_size}×{image_size} (will be resized to 448×448 during training)"
    )
    if stapled_ratio > 0:
        stapled_count = df["is_stapled"].sum()
        print(f"Stapled receipts: {stapled_count} ({stapled_count / num_collages:.1%})")

    return df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic receipt dataset")
    parser.add_argument(
        "--output_dir", default="synthetic_receipts", help="Output directory"
    )
    parser.add_argument(
        "--num_collages", type=int, default=300, help="Number of collages to generate"
    )
    parser.add_argument(
        "--count_probs",
        default="0.3,0.3,0.2,0.1,0.1,0",
        help="Probability distribution for receipt counts (0,1,2,3,4,5)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=2048,
        help="Size of output images (default: 2048 for high-resolution receipt photos)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--stapled_ratio",
        type=float,
        default=0.0,
        help="Ratio of images that should have stapled receipts (0.0-1.0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Parse probability distribution
    count_probs = [float(p) for p in args.count_probs.split(",")]

    # Generate dataset
    df = generate_dataset(
        output_dir=args.output_dir,
        num_collages=args.num_collages,
        count_probs=count_probs,
        image_size=args.image_size,
        stapled_ratio=args.stapled_ratio,
        seed=args.seed,
    )
