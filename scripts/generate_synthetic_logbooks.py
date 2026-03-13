#!/usr/bin/env python3
"""Generate synthetic vehicle logbook PNG images for testing document extraction."""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Output paths
OUTPUT_DIR = Path("/Users/tod/Desktop/LMM_POC/evaluation_data/travel")
CSV_PATH = OUTPUT_DIR / "ground_truth_logbook.csv"

# Australian vehicle data
MAKES_MODELS = [
    ("Toyota", "Corolla", "1.8L"),
    ("Toyota", "Camry", "2.5L"),
    ("Toyota", "HiLux", "2.8L"),
    ("Mazda", "3", "2.0L"),
    ("Mazda", "CX-5", "2.5L"),
    ("Ford", "Ranger", "3.2L"),
    ("Hyundai", "i30", "2.0L"),
    ("Kia", "Cerato", "2.0L"),
    ("Mitsubishi", "Triton", "2.4L"),
    ("Holden", "Colorado", "2.8L"),
]

STATE_PREFIXES = ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"]

BUSINESS_PURPOSES = [
    "Client meeting",
    "Site inspection",
    "Supplier visit",
    "Bank appointment",
    "Accountant meeting",
    "Delivery",
    "Sales call",
    "Training session",
    "Conference",
    "Project site",
    "Warehouse pickup",
    "Customer service call",
    "Office supplies run",
    "Equipment maintenance",
    "Staff training",
]

PERSONAL_PURPOSES = [
    "Personal",
    "Private use",
    "Shopping",
    "School pickup",
    "Medical appointment",
]


def generate_rego() -> str:
    """Generate Australian-style registration number."""
    state = random.choice(STATE_PREFIXES)
    if state in ["NSW", "VIC", "QLD"]:
        # Format: ABC-123 or AB-12-CD
        fmt = random.choice(["letters", "mixed"])
        if fmt == "letters":
            letters = "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=3))
            nums = "".join(random.choices("0123456789", k=3))
            return f"{letters}{nums}"
        else:
            l1 = "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=2))
            nums = "".join(random.choices("0123456789", k=2))
            l2 = "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=2))
            return f"{l1}{nums}{l2}"
    else:
        # Simpler format
        letters = "".join(random.choices("ABCDEFGHJKLMNPQRSTUVWXYZ", k=3))
        nums = "".join(random.choices("0123456789", k=3))
        return f"{letters}{nums}"


def generate_journeys(
    start_date: datetime, num_weeks: int, odometer_start: int
) -> list[dict]:
    """Generate realistic journey entries for the logbook period."""
    journeys = []
    current_odo = odometer_start
    current_date = start_date

    # Generate 3-6 trips per week on average
    for _week in range(num_weeks):
        trips_this_week = random.randint(2, 7)
        for _ in range(trips_this_week):
            # Random day within the week
            day_offset = random.randint(0, 6)
            trip_date = current_date + timedelta(days=day_offset)

            # 70-85% business use typically
            is_business = random.random() < 0.75
            purpose = (
                random.choice(BUSINESS_PURPOSES)
                if is_business
                else random.choice(PERSONAL_PURPOSES)
            )

            # Distance 5-120km per trip
            distance = random.randint(5, 120)

            journeys.append(
                {
                    "date": trip_date,
                    "odo_start": current_odo,
                    "odo_end": current_odo + distance,
                    "distance": distance,
                    "purpose": purpose,
                    "is_business": is_business,
                }
            )
            current_odo += distance

        current_date += timedelta(days=7)

    # Sort by date
    journeys.sort(key=lambda x: (x["date"], x["odo_start"]))
    return journeys


def format_date(dt: datetime) -> str:
    """Format date as DD Mon YYYY."""
    return dt.strftime("%d %b %Y")


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Get a font, falling back to default if needed."""
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


def get_handwriting_font(size: int) -> ImageFont.FreeTypeFont:
    """Get a handwriting-style font."""
    handwriting_paths = [
        "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
        "/System/Library/Fonts/Supplemental/Noteworthy.ttc",
        "/Library/Fonts/Comic Sans MS.ttf",
    ]
    for path in handwriting_paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return get_font(size)


def create_logbook_style_1(data: dict, filepath: Path):
    """Clean professional form style - Portrait."""
    width, height = 800, 1100
    img = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    font_title = get_font(28, bold=True)
    font_header = get_font(16, bold=True)
    font_normal = get_font(14)
    font_small = get_font(12)

    y = 30

    # Title
    draw.text((width // 2, y), "MOTOR VEHICLE LOGBOOK", fill="#000066", font=font_title, anchor="mm")
    y += 50

    # Vehicle details box
    draw.rectangle([40, y, width - 40, y + 120], outline="#000066", width=2)
    draw.text((50, y + 10), "VEHICLE DETAILS", fill="#000066", font=font_header)
    draw.text((50, y + 35), f"Make: {data['make']}", fill="#000000", font=font_normal)
    draw.text((250, y + 35), f"Model: {data['model']}", fill="#000000", font=font_normal)
    draw.text((450, y + 35), f"Registration: {data['rego']}", fill="#000000", font=font_normal)
    draw.text((50, y + 60), f"Engine Capacity: {data['engine']}", fill="#000000", font=font_normal)
    draw.text((50, y + 85), f"Logbook Period: {data['period_start']} to {data['period_end']}", fill="#000000", font=font_normal)
    y += 140

    # Odometer summary box
    draw.rectangle([40, y, width - 40, y + 80], outline="#000066", width=2)
    draw.text((50, y + 10), "ODOMETER READINGS", fill="#000066", font=font_header)
    draw.text((50, y + 35), f"Start: {data['odo_start']:,} km", fill="#000000", font=font_normal)
    draw.text((250, y + 35), f"End: {data['odo_end']:,} km", fill="#000000", font=font_normal)
    draw.text((450, y + 35), f"Total: {data['total_km']:,} km", fill="#000000", font=font_normal)
    draw.text((50, y + 55), f"Business: {data['business_km']:,} km ({data['business_pct']}%)", fill="#000000", font=font_normal)
    y += 100

    # Journey table header
    draw.rectangle([40, y, width - 40, y + 30], fill="#E6E6FA", outline="#000066", width=1)
    headers = ["Date", "Start", "End", "Km", "Purpose"]
    col_x = [50, 150, 230, 310, 380]
    for i, header in enumerate(headers):
        draw.text((col_x[i], y + 8), header, fill="#000066", font=font_header)
    y += 30

    # Journey entries (show first 15)
    for journey in data["journeys"][:15]:
        draw.line([(40, y + 22), (width - 40, y + 22)], fill="#CCCCCC", width=1)
        draw.text((col_x[0], y + 5), journey["date"].strftime("%d/%m"), fill="#000000", font=font_small)
        draw.text((col_x[1], y + 5), f"{journey['odo_start']:,}", fill="#000000", font=font_small)
        draw.text((col_x[2], y + 5), f"{journey['odo_end']:,}", fill="#000000", font=font_small)
        draw.text((col_x[3], y + 5), str(journey["distance"]), fill="#000000", font=font_small)
        purpose_text = journey["purpose"][:25] + "..." if len(journey["purpose"]) > 25 else journey["purpose"]
        color = "#000000" if journey["is_business"] else "#666666"
        draw.text((col_x[4], y + 5), purpose_text, fill=color, font=font_small)
        y += 24

    # Business use calculation box at bottom
    y = height - 120
    draw.rectangle([40, y, width - 40, y + 90], fill="#F0F8FF", outline="#000066", width=2)
    draw.text((50, y + 10), "BUSINESS USE CALCULATION", fill="#000066", font=font_header)
    draw.text((50, y + 40), f"Business Kilometres: {data['business_km']:,} km", fill="#000000", font=font_normal)
    draw.text((50, y + 60), f"Total Kilometres: {data['total_km']:,} km", fill="#000000", font=font_normal)
    draw.text((400, y + 50), f"Business Use: {data['business_pct']}%", fill="#000066", font=font_title)

    img.save(filepath, "PNG")


def create_logbook_style_2(data: dict, filepath: Path):
    """Spreadsheet/table style - Landscape."""
    width, height = 1200, 800
    img = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    font_title = get_font(24, bold=True)
    font_header = get_font(14, bold=True)
    font_normal = get_font(12)

    # Title row
    draw.text((width // 2, 25), "Vehicle Logbook - Tax Record", fill="#333333", font=font_title, anchor="mm")

    # Vehicle info row
    y = 55
    info_text = f"{data['make']} {data['model']} | Rego: {data['rego']} | Engine: {data['engine']} | Period: {data['period_start']} - {data['period_end']}"
    draw.text((width // 2, y), info_text, fill="#666666", font=font_normal, anchor="mm")

    # Table header
    y = 90
    draw.rectangle([20, y, width - 20, y + 28], fill="#4472C4", outline="#2F5597", width=1)
    headers = ["Date", "Odo Start", "Odo End", "Distance", "Business/Personal", "Purpose/Destination"]
    col_x = [30, 130, 240, 350, 440, 580]
    col_widths = [100, 110, 110, 90, 140, 400]
    for i, header in enumerate(headers):
        draw.text((col_x[i], y + 6), header, fill="#FFFFFF", font=font_header)
    y += 28

    # Alternating rows
    for idx, journey in enumerate(data["journeys"][:20]):
        bg_color = "#F2F2F2" if idx % 2 == 0 else "#FFFFFF"
        draw.rectangle([20, y, width - 20, y + 24], fill=bg_color)
        draw.text((col_x[0], y + 4), journey["date"].strftime("%d %b %Y"), fill="#000000", font=font_normal)
        draw.text((col_x[1], y + 4), f"{journey['odo_start']:,}", fill="#000000", font=font_normal)
        draw.text((col_x[2], y + 4), f"{journey['odo_end']:,}", fill="#000000", font=font_normal)
        draw.text((col_x[3], y + 4), f"{journey['distance']} km", fill="#000000", font=font_normal)
        bp_text = "Business" if journey["is_business"] else "Personal"
        bp_color = "#006600" if journey["is_business"] else "#990000"
        draw.text((col_x[4], y + 4), bp_text, fill=bp_color, font=font_normal)
        draw.text((col_x[5], y + 4), journey["purpose"], fill="#000000", font=font_normal)
        y += 24

    # Summary section
    y = height - 100
    draw.line([(20, y), (width - 20, y)], fill="#4472C4", width=2)
    y += 15
    draw.text((30, y), f"Odometer Start: {data['odo_start']:,} km", fill="#333333", font=font_normal)
    draw.text((230, y), f"Odometer End: {data['odo_end']:,} km", fill="#333333", font=font_normal)
    draw.text((430, y), f"Total Distance: {data['total_km']:,} km", fill="#333333", font=font_normal)
    y += 25
    draw.text((30, y), f"Business Kilometres: {data['business_km']:,} km", fill="#006600", font=font_header)
    draw.text((300, y), f"Personal Kilometres: {data['total_km'] - data['business_km']:,} km", fill="#990000", font=font_normal)
    draw.rectangle([550, y - 5, 750, y + 25], fill="#4472C4", outline="#2F5597", width=1)
    draw.text((650, y + 10), f"Business Use: {data['business_pct']}%", fill="#FFFFFF", font=font_header, anchor="mm")

    img.save(filepath, "PNG")


def create_logbook_style_3(data: dict, filepath: Path):
    """Handwritten style on lined paper."""
    width, height = 850, 1100
    img = Image.new("RGB", (width, height), "#FFFEF0")  # Cream paper
    draw = ImageDraw.Draw(img)

    # Draw lined paper
    for y_line in range(60, height, 28):
        draw.line([(40, y_line), (width - 40, y_line)], fill="#ADD8E6", width=1)

    # Red margin line
    draw.line([(80, 0), (80, height)], fill="#FFB6C1", width=1)

    font_title = get_handwriting_font(26)
    font_normal = get_handwriting_font(18)
    font_small = get_handwriting_font(14)

    y = 35
    draw.text((100, y), "Motor Vehicle Logbook", fill="#000080", font=font_title)
    y += 35

    draw.text((100, y), f"Vehicle: {data['make']} {data['model']}", fill="#00008B", font=font_normal)
    y += 28
    draw.text((100, y), f"Registration: {data['rego']}  Engine: {data['engine']}", fill="#00008B", font=font_normal)
    y += 28
    draw.text((100, y), f"Period: {data['period_start']} to {data['period_end']}", fill="#00008B", font=font_normal)
    y += 35

    draw.text((100, y), "Date        Odo Start   Odo End    Km    Purpose", fill="#000080", font=font_normal)
    y += 28

    for journey in data["journeys"][:18]:
        entry = f"{journey['date'].strftime('%d/%m')}        {journey['odo_start']}      {journey['odo_end']}      {journey['distance']}    {journey['purpose'][:20]}"
        ink_color = "#00008B" if journey["is_business"] else "#4B0082"
        draw.text((100, y), entry, fill=ink_color, font=font_small)
        y += 28

    # Summary at bottom
    y = height - 180
    draw.text((100, y), "Summary:", fill="#000080", font=font_title)
    y += 30
    draw.text((100, y), f"Starting Odometer: {data['odo_start']:,} km", fill="#00008B", font=font_normal)
    y += 28
    draw.text((100, y), f"Ending Odometer: {data['odo_end']:,} km", fill="#00008B", font=font_normal)
    y += 28
    draw.text((100, y), f"Total: {data['total_km']:,} km  Business: {data['business_km']:,} km", fill="#00008B", font=font_normal)
    y += 28
    draw.text((100, y), f"Business Use Percentage: {data['business_pct']}%", fill="#8B0000", font=font_title)

    img.save(filepath, "PNG")


def create_logbook_style_4(data: dict, filepath: Path):
    """Government/ATO form style."""
    width, height = 850, 1150
    img = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    font_title = get_font(22, bold=True)
    font_section = get_font(14, bold=True)
    font_normal = get_font(12)
    font_small = get_font(10)

    # Header with coat of arms style
    draw.rectangle([0, 0, width, 70], fill="#003366")
    draw.text((width // 2, 20), "AUSTRALIAN TAXATION OFFICE", fill="#FFFFFF", font=font_title, anchor="mm")
    draw.text((width // 2, 50), "Motor Vehicle Logbook Record", fill="#FFD700", font=font_section, anchor="mm")

    y = 90

    # Section A: Vehicle Details
    draw.rectangle([30, y, width - 30, y + 25], fill="#E8E8E8")
    draw.text((40, y + 5), "SECTION A: VEHICLE DETAILS", fill="#003366", font=font_section)
    y += 30

    # Form fields with boxes
    fields_a = [
        ("Make:", data["make"]),
        ("Model:", data["model"]),
        ("Registration Number:", data["rego"]),
        ("Engine Capacity:", data["engine"]),
    ]
    for label, value in fields_a:
        draw.text((40, y + 5), label, fill="#000000", font=font_normal)
        draw.rectangle([180, y, 350, y + 22], outline="#999999", width=1)
        draw.text((185, y + 4), value, fill="#000080", font=font_normal)
        y += 28

    y += 10

    # Section B: Logbook Period
    draw.rectangle([30, y, width - 30, y + 25], fill="#E8E8E8")
    draw.text((40, y + 5), "SECTION B: LOGBOOK PERIOD (minimum 12 continuous weeks)", fill="#003366", font=font_section)
    y += 30

    draw.text((40, y + 5), "Start Date:", fill="#000000", font=font_normal)
    draw.rectangle([120, y, 280, y + 22], outline="#999999", width=1)
    draw.text((125, y + 4), data["period_start"], fill="#000080", font=font_normal)
    draw.text((300, y + 5), "End Date:", fill="#000000", font=font_normal)
    draw.rectangle([380, y, 540, y + 22], outline="#999999", width=1)
    draw.text((385, y + 4), data["period_end"], fill="#000080", font=font_normal)
    y += 35

    draw.text((40, y + 5), "Odometer at Start:", fill="#000000", font=font_normal)
    draw.rectangle([160, y, 280, y + 22], outline="#999999", width=1)
    draw.text((165, y + 4), f"{data['odo_start']:,}", fill="#000080", font=font_normal)
    draw.text((300, y + 5), "Odometer at End:", fill="#000000", font=font_normal)
    draw.rectangle([420, y, 540, y + 22], outline="#999999", width=1)
    draw.text((425, y + 4), f"{data['odo_end']:,}", fill="#000080", font=font_normal)
    y += 40

    # Section C: Journey Record
    draw.rectangle([30, y, width - 30, y + 25], fill="#E8E8E8")
    draw.text((40, y + 5), "SECTION C: JOURNEY RECORD", fill="#003366", font=font_section)
    y += 30

    # Table header
    draw.rectangle([30, y, width - 30, y + 22], fill="#CCCCCC", outline="#999999")
    th = ["Date", "Odometer Start", "Odometer End", "Km", "Business?", "Purpose"]
    tx = [40, 130, 260, 380, 440, 520]
    for i, h in enumerate(th):
        draw.text((tx[i], y + 4), h, fill="#000000", font=font_small)
    y += 22

    for journey in data["journeys"][:12]:
        draw.rectangle([30, y, width - 30, y + 20], outline="#DDDDDD")
        draw.text((tx[0], y + 3), journey["date"].strftime("%d/%m/%y"), fill="#000000", font=font_small)
        draw.text((tx[1], y + 3), f"{journey['odo_start']:,}", fill="#000000", font=font_small)
        draw.text((tx[2], y + 3), f"{journey['odo_end']:,}", fill="#000000", font=font_small)
        draw.text((tx[3], y + 3), str(journey["distance"]), fill="#000000", font=font_small)
        yn = "Yes" if journey["is_business"] else "No"
        draw.text((tx[4], y + 3), yn, fill="#000000", font=font_small)
        draw.text((tx[5], y + 3), journey["purpose"][:22], fill="#000000", font=font_small)
        y += 20

    # Section D: Calculation
    y = height - 150
    draw.rectangle([30, y, width - 30, y + 25], fill="#E8E8E8")
    draw.text((40, y + 5), "SECTION D: BUSINESS USE CALCULATION", fill="#003366", font=font_section)
    y += 35

    draw.text((40, y), f"Total Kilometres Travelled: {data['total_km']:,} km", fill="#000000", font=font_normal)
    y += 25
    draw.text((40, y), f"Business Kilometres: {data['business_km']:,} km", fill="#000000", font=font_normal)
    y += 25
    draw.rectangle([40, y, 300, y + 35], fill="#003366")
    draw.text((170, y + 17), f"BUSINESS USE: {data['business_pct']}%", fill="#FFFFFF", font=font_title, anchor="mm")

    img.save(filepath, "PNG")


def create_logbook_style_5(data: dict, filepath: Path):
    """Simple notebook/diary style."""
    width, height = 750, 1000
    img = Image.new("RGB", (width, height), "#F5F5DC")  # Beige
    draw = ImageDraw.Draw(img)

    font_title = get_font(24, bold=True)
    font_normal = get_font(14)
    font_small = get_font(11)

    # Spiral binding effect
    for sy in range(50, height, 40):
        draw.ellipse([10, sy - 5, 25, sy + 5], fill="#888888", outline="#666666")

    y = 40
    draw.text((50, y), "LOGBOOK", fill="#8B4513", font=font_title)
    y += 35

    draw.text((50, y), f"{data['make']} {data['model']} - {data['rego']}", fill="#333333", font=font_normal)
    y += 25
    draw.text((50, y), f"Engine: {data['engine']}", fill="#333333", font=font_normal)
    y += 35

    draw.text((50, y), f"Period: {data['period_start']} to {data['period_end']}", fill="#8B4513", font=font_normal)
    y += 25
    draw.text((50, y), f"Start Odo: {data['odo_start']:,}   End Odo: {data['odo_end']:,}", fill="#333333", font=font_normal)
    y += 40

    # Simple table
    draw.line([(50, y), (700, y)], fill="#8B4513", width=2)
    y += 5
    draw.text((50, y), "Date", fill="#8B4513", font=font_normal)
    draw.text((150, y), "From", fill="#8B4513", font=font_normal)
    draw.text((230, y), "To", fill="#8B4513", font=font_normal)
    draw.text((310, y), "Km", fill="#8B4513", font=font_normal)
    draw.text((370, y), "Type", fill="#8B4513", font=font_normal)
    draw.text((450, y), "Purpose", fill="#8B4513", font=font_normal)
    y += 20
    draw.line([(50, y), (700, y)], fill="#8B4513", width=1)
    y += 8

    for journey in data["journeys"][:20]:
        draw.text((50, y), journey["date"].strftime("%d %b"), fill="#000000", font=font_small)
        draw.text((150, y), str(journey["odo_start"]), fill="#000000", font=font_small)
        draw.text((230, y), str(journey["odo_end"]), fill="#000000", font=font_small)
        draw.text((310, y), str(journey["distance"]), fill="#000000", font=font_small)
        t = "B" if journey["is_business"] else "P"
        draw.text((370, y), t, fill="#006400" if t == "B" else "#8B0000", font=font_small)
        draw.text((450, y), journey["purpose"][:25], fill="#000000", font=font_small)
        y += 22

    # Totals
    y = height - 120
    draw.line([(50, y), (700, y)], fill="#8B4513", width=2)
    y += 15
    draw.text((50, y), f"TOTAL KM: {data['total_km']:,}", fill="#333333", font=font_normal)
    draw.text((250, y), f"BUSINESS KM: {data['business_km']:,}", fill="#006400", font=font_normal)
    y += 30
    draw.text((50, y), f"BUSINESS USE PERCENTAGE: {data['business_pct']}%", fill="#8B4513", font=font_title)

    img.save(filepath, "PNG")


def create_logbook_style_6(data: dict, filepath: Path):
    """Commercial logbook template style with logo placeholder."""
    width, height = 900, 1200
    img = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    font_title = get_font(26, bold=True)
    font_subtitle = get_font(16, bold=True)
    font_normal = get_font(13)
    font_small = get_font(11)

    # Header with "company" branding
    draw.rectangle([0, 0, width, 80], fill="#2E8B57")
    draw.text((width // 2, 25), "PROFESSIONAL MOTOR VEHICLE LOGBOOK", fill="#FFFFFF", font=font_title, anchor="mm")
    draw.text((width // 2, 55), "ATO Compliant Record for Tax Deduction Claims", fill="#90EE90", font=font_normal, anchor="mm")

    y = 100

    # Two-column vehicle info
    draw.rectangle([30, y, 430, y + 100], outline="#2E8B57", width=2)
    draw.text((40, y + 10), "Vehicle Information", fill="#2E8B57", font=font_subtitle)
    draw.text((40, y + 35), f"Make: {data['make']}", fill="#000000", font=font_normal)
    draw.text((40, y + 55), f"Model: {data['model']}", fill="#000000", font=font_normal)
    draw.text((40, y + 75), f"Engine: {data['engine']}", fill="#000000", font=font_normal)

    draw.rectangle([460, y, 870, y + 100], outline="#2E8B57", width=2)
    draw.text((470, y + 10), "Registration & Period", fill="#2E8B57", font=font_subtitle)
    draw.text((470, y + 35), f"Rego: {data['rego']}", fill="#000000", font=font_normal)
    draw.text((470, y + 55), f"From: {data['period_start']}", fill="#000000", font=font_normal)
    draw.text((470, y + 75), f"To: {data['period_end']}", fill="#000000", font=font_normal)

    y += 120

    # Odometer summary
    draw.rectangle([30, y, 870, y + 50], fill="#F0FFF0", outline="#2E8B57", width=1)
    draw.text((50, y + 15), f"Odometer Start: {data['odo_start']:,} km", fill="#000000", font=font_normal)
    draw.text((300, y + 15), f"Odometer End: {data['odo_end']:,} km", fill="#000000", font=font_normal)
    draw.text((550, y + 15), f"Total Distance: {data['total_km']:,} km", fill="#2E8B57", font=font_subtitle)
    y += 70

    # Journey log header
    draw.rectangle([30, y, 870, y + 30], fill="#2E8B57")
    headers = ["Date", "Odo Start", "Odo End", "Distance", "B/P", "Journey Purpose"]
    hx = [40, 130, 230, 340, 430, 500]
    for i, h in enumerate(headers):
        draw.text((hx[i], y + 7), h, fill="#FFFFFF", font=font_small)
    y += 30

    # Journey entries
    for idx, journey in enumerate(data["journeys"][:18]):
        bg = "#F0FFF0" if idx % 2 == 0 else "#FFFFFF"
        draw.rectangle([30, y, 870, y + 25], fill=bg)
        draw.text((hx[0], y + 5), journey["date"].strftime("%d/%m/%Y"), fill="#000000", font=font_small)
        draw.text((hx[1], y + 5), f"{journey['odo_start']:,}", fill="#000000", font=font_small)
        draw.text((hx[2], y + 5), f"{journey['odo_end']:,}", fill="#000000", font=font_small)
        draw.text((hx[3], y + 5), f"{journey['distance']} km", fill="#000000", font=font_small)
        bp = "Business" if journey["is_business"] else "Personal"
        bp_color = "#006400" if journey["is_business"] else "#8B0000"
        draw.text((hx[4], y + 5), bp, fill=bp_color, font=font_small)
        draw.text((hx[5], y + 5), journey["purpose"][:35], fill="#000000", font=font_small)
        y += 25

    # Summary box
    y = height - 130
    draw.rectangle([30, y, 870, y + 100], fill="#2E8B57")
    draw.text((50, y + 15), "SUMMARY", fill="#FFFFFF", font=font_title)
    draw.text((50, y + 50), f"Business Kilometres: {data['business_km']:,} km", fill="#FFFFFF", font=font_normal)
    draw.text((50, y + 75), f"Personal Kilometres: {data['total_km'] - data['business_km']:,} km", fill="#FFFFFF", font=font_normal)
    draw.rectangle([550, y + 30, 850, y + 85], fill="#FFFFFF")
    draw.text((700, y + 57), f"Business Use: {data['business_pct']}%", fill="#2E8B57", font=font_title, anchor="mm")

    img.save(filepath, "PNG")


def create_logbook_style_7(data: dict, filepath: Path):
    """Minimal clean modern style."""
    width, height = 800, 1000
    img = Image.new("RGB", (width, height), "#FAFAFA")
    draw = ImageDraw.Draw(img)

    font_title = get_font(32, bold=True)
    font_section = get_font(18, bold=True)
    font_normal = get_font(14)
    font_small = get_font(12)

    y = 50
    draw.text((50, y), "Vehicle Logbook", fill="#1A1A1A", font=font_title)
    y += 50

    # Minimal info line
    draw.text((50, y), f"{data['make']} {data['model']}  |  {data['rego']}  |  {data['engine']}", fill="#666666", font=font_normal)
    y += 30
    draw.text((50, y), f"{data['period_start']} — {data['period_end']}", fill="#999999", font=font_normal)
    y += 50

    # Key metrics
    draw.rectangle([50, y, 250, y + 80], fill="#FFFFFF", outline="#E0E0E0", width=1)
    draw.text((150, y + 15), f"{data['odo_start']:,}", fill="#1A1A1A", font=font_section, anchor="mm")
    draw.text((150, y + 50), "Start Odometer", fill="#999999", font=font_small, anchor="mm")

    draw.rectangle([280, y, 480, y + 80], fill="#FFFFFF", outline="#E0E0E0", width=1)
    draw.text((380, y + 15), f"{data['odo_end']:,}", fill="#1A1A1A", font=font_section, anchor="mm")
    draw.text((380, y + 50), "End Odometer", fill="#999999", font=font_small, anchor="mm")

    draw.rectangle([510, y, 750, y + 80], fill="#1A1A1A")
    draw.text((630, y + 15), f"{data['business_pct']}%", fill="#FFFFFF", font=font_title, anchor="mm")
    draw.text((630, y + 55), "Business Use", fill="#CCCCCC", font=font_small, anchor="mm")

    y += 110

    # Simple journey list
    draw.text((50, y), "Trip Log", fill="#1A1A1A", font=font_section)
    y += 35

    for journey in data["journeys"][:16]:
        date_str = journey["date"].strftime("%d %b")
        km_str = f"{journey['distance']} km"
        purpose = journey["purpose"]
        b_marker = "●" if journey["is_business"] else "○"
        marker_color = "#1A1A1A" if journey["is_business"] else "#CCCCCC"

        draw.text((50, y), b_marker, fill=marker_color, font=font_normal)
        draw.text((75, y), date_str, fill="#666666", font=font_small)
        draw.text((160, y), km_str, fill="#1A1A1A", font=font_small)
        draw.text((240, y), purpose, fill="#333333", font=font_small)
        y += 26

    # Footer stats
    y = height - 100
    draw.line([(50, y), (750, y)], fill="#E0E0E0", width=1)
    y += 20
    draw.text((50, y), f"Total: {data['total_km']:,} km", fill="#666666", font=font_normal)
    draw.text((250, y), f"Business: {data['business_km']:,} km", fill="#1A1A1A", font=font_normal)
    draw.text((500, y), f"Personal: {data['total_km'] - data['business_km']:,} km", fill="#999999", font=font_normal)

    img.save(filepath, "PNG")


def create_logbook_style_8(data: dict, filepath: Path):
    """Detailed multi-column accounting style."""
    width, height = 1100, 850
    img = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    font_title = get_font(20, bold=True)
    font_header = get_font(11, bold=True)
    font_normal = get_font(10)
    font_tiny = get_font(9)

    # Top info bar
    draw.rectangle([0, 0, width, 60], fill="#333333")
    draw.text((20, 20), "MOTOR VEHICLE LOGBOOK", fill="#FFFFFF", font=font_title)
    draw.text((width - 20, 20), f"Business Use: {data['business_pct']}%", fill="#FFD700", font=font_title, anchor="ra")

    y = 75

    # Vehicle and period info
    info_parts = [
        f"Vehicle: {data['make']} {data['model']}",
        f"Rego: {data['rego']}",
        f"Engine: {data['engine']}",
        f"Period: {data['period_start']} to {data['period_end']}",
        f"Odo Start: {data['odo_start']:,}",
        f"Odo End: {data['odo_end']:,}",
    ]
    x_pos = 20
    for info in info_parts:
        draw.text((x_pos, y), info, fill="#333333", font=font_normal)
        x_pos += 175

    y += 35

    # Detailed table header
    draw.rectangle([20, y, width - 20, y + 25], fill="#4A4A4A")
    cols = ["#", "Date", "Day", "Odo Start", "Odo End", "Total Km", "Business Km", "Personal Km", "Purpose/Destination"]
    cx = [30, 60, 150, 210, 300, 390, 480, 580, 680]
    for i, col in enumerate(cols):
        draw.text((cx[i], y + 6), col, fill="#FFFFFF", font=font_header)
    y += 25

    # Entries with running totals
    running_bus = 0
    running_per = 0
    for idx, journey in enumerate(data["journeys"][:22], 1):
        bg = "#F5F5F5" if idx % 2 == 0 else "#FFFFFF"
        draw.rectangle([20, y, width - 20, y + 22], fill=bg)

        bus_km = journey["distance"] if journey["is_business"] else 0
        per_km = 0 if journey["is_business"] else journey["distance"]
        running_bus += bus_km
        running_per += per_km

        draw.text((cx[0], y + 4), str(idx), fill="#666666", font=font_tiny)
        draw.text((cx[1], y + 4), journey["date"].strftime("%d/%m/%Y"), fill="#000000", font=font_tiny)
        draw.text((cx[2], y + 4), journey["date"].strftime("%A")[:3], fill="#666666", font=font_tiny)
        draw.text((cx[3], y + 4), f"{journey['odo_start']:,}", fill="#000000", font=font_tiny)
        draw.text((cx[4], y + 4), f"{journey['odo_end']:,}", fill="#000000", font=font_tiny)
        draw.text((cx[5], y + 4), str(journey["distance"]), fill="#000000", font=font_tiny)
        draw.text((cx[6], y + 4), str(bus_km) if bus_km else "-", fill="#006600" if bus_km else "#CCCCCC", font=font_tiny)
        draw.text((cx[7], y + 4), str(per_km) if per_km else "-", fill="#990000" if per_km else "#CCCCCC", font=font_tiny)
        draw.text((cx[8], y + 4), journey["purpose"][:40], fill="#333333", font=font_tiny)
        y += 22

    # Totals row
    y += 5
    draw.rectangle([20, y, width - 20, y + 28], fill="#333333")
    draw.text((cx[4], y + 6), "TOTALS:", fill="#FFFFFF", font=font_header)
    draw.text((cx[5], y + 6), f"{data['total_km']:,}", fill="#FFFFFF", font=font_header)
    draw.text((cx[6], y + 6), f"{data['business_km']:,}", fill="#00FF00", font=font_header)
    draw.text((cx[7], y + 6), f"{data['total_km'] - data['business_km']:,}", fill="#FF6666", font=font_header)

    img.save(filepath, "PNG")


def create_logbook_style_9(data: dict, filepath: Path):
    """Scanned/aged document style."""
    width, height = 850, 1100
    # Slightly off-white, aged paper look
    img = Image.new("RGB", (width, height), "#F8F4E8")
    draw = ImageDraw.Draw(img)

    # Add some "scan artifacts" - slight discoloration
    for _ in range(20):
        x = random.randint(0, width)
        y_spot = random.randint(0, height)
        r = random.randint(5, 30)
        shade = random.randint(230, 245)
        draw.ellipse([x - r, y_spot - r, x + r, y_spot + r], fill=(shade, shade - 5, shade - 10))

    font_title = get_font(22, bold=True)
    font_normal = get_font(13)
    font_small = get_font(11)

    y = 40
    draw.text((width // 2, y), "VEHICLE LOGBOOK", fill="#1A1A1A", font=font_title, anchor="mm")
    y += 40

    # Hand-drawn style boxes
    draw.rectangle([40, y, width - 40, y + 90], outline="#444444", width=1)
    draw.text((50, y + 10), f"Make/Model: {data['make']} {data['model']}", fill="#222222", font=font_normal)
    draw.text((50, y + 32), f"Registration: {data['rego']}", fill="#222222", font=font_normal)
    draw.text((350, y + 32), f"Engine: {data['engine']}", fill="#222222", font=font_normal)
    draw.text((50, y + 54), f"Logbook Period: {data['period_start']} - {data['period_end']}", fill="#222222", font=font_normal)
    y += 110

    draw.text((50, y), f"Opening Odometer: {data['odo_start']:,} km", fill="#222222", font=font_normal)
    draw.text((400, y), f"Closing Odometer: {data['odo_end']:,} km", fill="#222222", font=font_normal)
    y += 35

    # Table with slightly uneven lines
    draw.line([(40, y), (width - 40, y)], fill="#333333", width=2)
    y += 5
    headers = ["Date", "Start", "End", "Km", "Bus.", "Purpose"]
    hx = [50, 150, 250, 350, 420, 500]
    for i, h in enumerate(headers):
        draw.text((hx[i], y), h, fill="#333333", font=font_normal)
    y += 22
    draw.line([(40, y), (width - 40, y)], fill="#666666", width=1)
    y += 5

    for journey in data["journeys"][:17]:
        # Slight position variation for "scanned" look
        x_offset = random.randint(-1, 1)
        draw.text((hx[0] + x_offset, y), journey["date"].strftime("%d/%m/%y"), fill="#1A1A1A", font=font_small)
        draw.text((hx[1] + x_offset, y), str(journey["odo_start"]), fill="#1A1A1A", font=font_small)
        draw.text((hx[2] + x_offset, y), str(journey["odo_end"]), fill="#1A1A1A", font=font_small)
        draw.text((hx[3] + x_offset, y), str(journey["distance"]), fill="#1A1A1A", font=font_small)
        yn = "Y" if journey["is_business"] else "N"
        draw.text((hx[4] + x_offset, y), yn, fill="#1A1A1A", font=font_small)
        draw.text((hx[5] + x_offset, y), journey["purpose"][:28], fill="#1A1A1A", font=font_small)
        y += 24

    # Summary at bottom
    y = height - 160
    draw.line([(40, y), (width - 40, y)], fill="#333333", width=2)
    y += 15
    draw.text((50, y), "SUMMARY", fill="#333333", font=font_title)
    y += 35
    draw.text((50, y), f"Total Kilometres: {data['total_km']:,}", fill="#222222", font=font_normal)
    y += 25
    draw.text((50, y), f"Business Kilometres: {data['business_km']:,}", fill="#222222", font=font_normal)
    y += 25
    draw.text((50, y), f"Business Use Percentage: {data['business_pct']}%", fill="#000000", font=font_title)

    img.save(filepath, "PNG")


def create_logbook_style_10(data: dict, filepath: Path):
    """Mobile app screenshot style."""
    width, height = 450, 900
    img = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(img)

    font_title = get_font(22, bold=True)
    font_section = get_font(16, bold=True)
    font_normal = get_font(14)
    font_small = get_font(12)

    # Status bar
    draw.rectangle([0, 0, width, 35], fill="#007AFF")
    draw.text((width // 2, 18), "myLogbook", fill="#FFFFFF", font=font_normal, anchor="mm")

    y = 50

    # Vehicle card
    draw.rectangle([15, y, width - 15, y + 100], fill="#F2F2F7", outline=None)
    draw.rounded_rectangle([15, y, width - 15, y + 100], radius=12, fill="#F2F2F7")
    draw.text((30, y + 15), f"{data['make']} {data['model']}", fill="#000000", font=font_section)
    draw.text((30, y + 40), f"{data['rego']}  •  {data['engine']}", fill="#666666", font=font_normal)
    draw.text((30, y + 65), f"{data['period_start']} – {data['period_end']}", fill="#999999", font=font_small)
    y += 120

    # Stats cards
    stats = [
        ("Total", f"{data['total_km']:,} km", "#333333"),
        ("Business", f"{data['business_km']:,} km", "#34C759"),
        ("% Business", f"{data['business_pct']}%", "#007AFF"),
    ]
    card_width = (width - 60) // 3
    for i, (label, value, color) in enumerate(stats):
        cx = 20 + i * (card_width + 10)
        draw.rounded_rectangle([cx, y, cx + card_width, y + 70], radius=8, fill="#F2F2F7")
        draw.text((cx + card_width // 2, y + 20), value, fill=color, font=font_section, anchor="mm")
        draw.text((cx + card_width // 2, y + 50), label, fill="#666666", font=font_small, anchor="mm")
    y += 90

    # Odometer
    draw.text((20, y), "Odometer", fill="#666666", font=font_small)
    y += 20
    draw.text((20, y), f"Start: {data['odo_start']:,}", fill="#333333", font=font_normal)
    draw.text((width // 2, y), f"End: {data['odo_end']:,}", fill="#333333", font=font_normal)
    y += 35

    # Recent trips
    draw.text((20, y), "Recent Trips", fill="#000000", font=font_section)
    y += 30

    for journey in data["journeys"][:10]:
        draw.rounded_rectangle([15, y, width - 15, y + 55], radius=8, fill="#F2F2F7")
        draw.text((30, y + 10), journey["date"].strftime("%d %b"), fill="#333333", font=font_normal)
        draw.text((120, y + 10), f"{journey['distance']} km", fill="#333333", font=font_normal)
        indicator = "●" if journey["is_business"] else "○"
        ind_color = "#34C759" if journey["is_business"] else "#FF3B30"
        draw.text((width - 40, y + 10), indicator, fill=ind_color, font=font_normal)
        draw.text((30, y + 32), journey["purpose"][:35], fill="#666666", font=font_small)
        y += 62

    img.save(filepath, "PNG")


def generate_logbook_data(index: int) -> dict:
    """Generate realistic logbook data for one vehicle."""
    make, model, engine = random.choice(MAKES_MODELS)
    rego = generate_rego()

    # 12-week logbook period (ATO minimum)
    year = 2025
    start_month = random.randint(1, 9)  # Allow room for 12 weeks
    period_start = datetime(year, start_month, 1)
    period_end = period_start + timedelta(weeks=12) - timedelta(days=1)

    # Starting odometer 10,000 - 150,000
    odo_start = random.randint(10000, 150000)

    # Generate journeys
    journeys = generate_journeys(period_start, 12, odo_start)

    # Calculate totals
    total_km = sum(j["distance"] for j in journeys)
    business_km = sum(j["distance"] for j in journeys if j["is_business"])
    odo_end = odo_start + total_km
    business_pct = round((business_km / total_km) * 100) if total_km > 0 else 0

    return {
        "make": make,
        "model": model,
        "engine": engine,
        "rego": rego,
        "period_start": format_date(period_start),
        "period_end": format_date(period_end),
        "odo_start": odo_start,
        "odo_end": odo_end,
        "total_km": total_km,
        "business_km": business_km,
        "business_pct": business_pct,
        "journeys": journeys,
    }


def data_to_csv_row(filename: str, data: dict) -> dict:
    """Convert logbook data to CSV row format."""
    # Extract journey details for list fields (first 10 journeys)
    sample_journeys = data["journeys"][:10]
    journey_dates = " | ".join(format_date(j["date"]) for j in sample_journeys)
    journey_distances = " | ".join(str(j["distance"]) for j in sample_journeys)
    journey_purposes = " | ".join(j["purpose"] for j in sample_journeys)

    return {
        "filename": filename,
        "DOCUMENT_TYPE": "VEHICLE_LOGBOOK",
        "VEHICLE_MAKE": data["make"],
        "VEHICLE_MODEL": data["model"],
        "VEHICLE_REGISTRATION": data["rego"],
        "ENGINE_CAPACITY": data["engine"],
        "LOGBOOK_PERIOD_START": data["period_start"],
        "LOGBOOK_PERIOD_END": data["period_end"],
        "ODOMETER_START": str(data["odo_start"]),
        "ODOMETER_END": str(data["odo_end"]),
        "TOTAL_KILOMETERS": str(data["total_km"]),
        "BUSINESS_KILOMETERS": str(data["business_km"]),
        "BUSINESS_USE_PERCENTAGE": f"{data['business_pct']}%",
        "JOURNEY_DATES": journey_dates,
        "JOURNEY_DISTANCES": journey_distances,
        "JOURNEY_PURPOSES": journey_purposes,
    }


def main():
    """Generate 10 synthetic logbook images and CSV."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Style functions
    styles = [
        ("logbook_001_professional.png", create_logbook_style_1),
        ("logbook_002_spreadsheet.png", create_logbook_style_2),
        ("logbook_003_handwritten.png", create_logbook_style_3),
        ("logbook_004_ato_form.png", create_logbook_style_4),
        ("logbook_005_notebook.png", create_logbook_style_5),
        ("logbook_006_commercial.png", create_logbook_style_6),
        ("logbook_007_minimal.png", create_logbook_style_7),
        ("logbook_008_accounting.png", create_logbook_style_8),
        ("logbook_009_scanned.png", create_logbook_style_9),
        ("logbook_010_mobile_app.png", create_logbook_style_10),
    ]

    csv_rows = []
    random.seed(42)  # Reproducibility

    for i, (filename, style_func) in enumerate(styles):
        print(f"Generating {filename}...")
        data = generate_logbook_data(i)
        filepath = OUTPUT_DIR / filename
        style_func(data, filepath)
        csv_rows.append(data_to_csv_row(filename, data))

    # Write CSV
    fieldnames = [
        "filename",
        "DOCUMENT_TYPE",
        "VEHICLE_MAKE",
        "VEHICLE_MODEL",
        "VEHICLE_REGISTRATION",
        "ENGINE_CAPACITY",
        "LOGBOOK_PERIOD_START",
        "LOGBOOK_PERIOD_END",
        "ODOMETER_START",
        "ODOMETER_END",
        "TOTAL_KILOMETERS",
        "BUSINESS_KILOMETERS",
        "BUSINESS_USE_PERCENTAGE",
        "JOURNEY_DATES",
        "JOURNEY_DISTANCES",
        "JOURNEY_PURPOSES",
    ]

    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nGenerated 10 logbook images in {OUTPUT_DIR}")
    print(f"Ground truth CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
