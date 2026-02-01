"""
Generate synthetic airline ticket images for document extraction testing.

Uses fictional Australian airlines to avoid trademark issues.
Generates tickets for travel between Australian state capital cities.
"""

import random
import string
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Fictional Australian airlines
AIRLINES = [
    {"name": "Southern Cross Airways", "code": "SX", "color": "#1E3A5F", "accent": "#D4AF37"},
    {"name": "Aussie Sky Airlines", "code": "AS", "color": "#006B3C", "accent": "#FFD700"},
    {"name": "Outback Air", "code": "OA", "color": "#8B4513", "accent": "#F4A460"},
    {"name": "Pacific Wanderer", "code": "PW", "color": "#4169E1", "accent": "#87CEEB"},
]

# Australian state capital cities with airport codes
CITIES = [
    {"city": "Sydney", "state": "NSW", "code": "SYD"},
    {"city": "Melbourne", "state": "VIC", "code": "MEL"},
    {"city": "Brisbane", "state": "QLD", "code": "BNE"},
    {"city": "Perth", "state": "WA", "code": "PER"},
    {"city": "Adelaide", "state": "SA", "code": "ADL"},
    {"city": "Hobart", "state": "TAS", "code": "HBA"},
    {"city": "Darwin", "state": "NT", "code": "DRW"},
    {"city": "Canberra", "state": "ACT", "code": "CBR"},
]

# Passenger name pools
FIRST_NAMES = [
    "James", "Emma", "Oliver", "Charlotte", "William", "Amelia", "Jack", "Olivia",
    "Noah", "Ava", "Thomas", "Mia", "Henry", "Isabella", "Leo", "Sophie",
    "Alexander", "Grace", "Lucas", "Chloe", "Ethan", "Lily", "Benjamin", "Ella",
]

LAST_NAMES = [
    "Smith", "Jones", "Williams", "Brown", "Wilson", "Taylor", "Johnson", "White",
    "Martin", "Anderson", "Thompson", "Walker", "Harris", "Clark", "Robinson", "King",
    "Wright", "Scott", "Green", "Baker", "Hill", "Moore", "Lee", "Mitchell",
]

CLASSES = ["Economy", "Premium Economy", "Business"]
CLASS_WEIGHTS = [0.7, 0.2, 0.1]


def generate_booking_ref() -> str:
    """Generate a 6-character alphanumeric booking reference."""
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def generate_flight_number(airline_code: str) -> str:
    """Generate a flight number like SX1234."""
    return f"{airline_code}{random.randint(100, 999)}"


def generate_passenger_name() -> str:
    """Generate a random passenger name."""
    return f"{random.choice(LAST_NAMES)}/{random.choice(FIRST_NAMES)}"


def generate_seat() -> str:
    """Generate a seat assignment like 14A."""
    row = random.randint(1, 35)
    seat = random.choice("ABCDEF")
    return f"{row}{seat}"


def generate_gate() -> str:
    """Generate a gate number."""
    return f"{random.randint(1, 50)}"


def generate_price(travel_class: str) -> float:
    """Generate a realistic price based on class."""
    base_prices = {"Economy": (150, 450), "Premium Economy": (400, 800), "Business": (800, 2500)}
    low, high = base_prices[travel_class]
    return round(random.uniform(low, high), 2)


def calculate_gst(total_amount: float, gst_free: bool = False) -> tuple[float, float]:
    """
    Calculate GST from a GST-inclusive total.

    Australian domestic flights include 10% GST.
    International flights are GST-free.

    Returns:
        Tuple of (gst_amount, gst_free_subtotal)
    """
    if gst_free:
        return 0.0, total_amount

    # GST = Total / 11 (since Total = Subtotal + 10% of Subtotal = 1.1 * Subtotal)
    gst_amount = round(total_amount / 11, 2)
    subtotal = round(total_amount - gst_amount, 2)
    return gst_amount, subtotal


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def load_fonts():
    """Load fonts with fallbacks for different systems."""
    try:
        return {
            "large": ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32),
            "medium": ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20),
            "small": ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14),
            "tiny": ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11),
            "code": ImageFont.truetype("/System/Library/Fonts/Courier.dfont", 28),
            "code_small": ImageFont.truetype("/System/Library/Fonts/Courier.dfont", 18),
        }
    except (OSError, IOError):
        try:
            return {
                "large": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32),
                "medium": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20),
                "small": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14),
                "tiny": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11),
                "code": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 28),
                "code_small": ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 18),
            }
        except (OSError, IOError):
            default = ImageFont.load_default()
            return {
                "large": default,
                "medium": default,
                "small": default,
                "tiny": default,
                "code": default,
                "code_small": default,
            }


def draw_ticket(
    airline: dict,
    origin: dict,
    destination: dict,
    passenger: str,
    flight_date: datetime,
    flight_number: str,
    booking_ref: str,
    seat: str,
    gate: str,
    travel_class: str,
    price: float,
    output_path: Path,
    issue_date: datetime | None = None,
    gst_amount: float = 0.0,
) -> None:
    """Draw and save a single-leg airline ticket image."""

    # Ticket dimensions (standard boarding pass proportions) - increased height for GST info
    width, height = 900, 450
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Colors
    main_color = hex_to_rgb(airline["color"])
    accent_color = hex_to_rgb(airline["accent"])

    fonts = load_fonts()

    # Header bar with airline branding
    draw.rectangle([(0, 0), (width, 70)], fill=main_color)
    draw.text((20, 15), airline["name"].upper(), fill="white", font=fonts["large"])
    draw.text((width - 120, 25), f"FLIGHT {flight_number}", fill=accent_color, font=fonts["medium"])

    # Boarding pass label
    draw.rectangle([(width - 200, 70), (width, 110)], fill=accent_color)
    draw.text((width - 180, 78), "BOARDING PASS", fill=main_color, font=fonts["small"])

    # Main content area
    y_offset = 90

    # Route section (large airport codes)
    draw.text((30, y_offset + 20), origin["code"], fill=main_color, font=fonts["code"])
    draw.text((30, y_offset + 55), origin["city"], fill="gray", font=fonts["small"])

    # Arrow between cities
    draw.text((130, y_offset + 25), "→", fill=accent_color, font=fonts["large"])

    draw.text((180, y_offset + 20), destination["code"], fill=main_color, font=fonts["code"])
    draw.text((180, y_offset + 55), destination["city"], fill="gray", font=fonts["small"])

    # Passenger name
    draw.text((30, y_offset + 100), "PASSENGER", fill="gray", font=fonts["small"])
    draw.text((30, y_offset + 118), passenger, fill="black", font=fonts["medium"])

    # Flight details grid
    col2_x = 300
    col3_x = 450
    col4_x = 600
    col5_x = 750

    # Row 1: Date, Time, Gate, Seat
    draw.text((col2_x, y_offset + 20), "DATE", fill="gray", font=fonts["small"])
    draw.text((col2_x, y_offset + 38), flight_date.strftime("%d %b %Y"), fill="black", font=fonts["medium"])

    draw.text((col3_x, y_offset + 20), "DEPARTURE", fill="gray", font=fonts["small"])
    dep_time = flight_date.strftime("%H:%M")
    draw.text((col3_x, y_offset + 38), dep_time, fill="black", font=fonts["medium"])

    draw.text((col4_x, y_offset + 20), "GATE", fill="gray", font=fonts["small"])
    draw.text((col4_x, y_offset + 38), gate, fill="black", font=fonts["medium"])

    draw.text((col5_x, y_offset + 20), "SEAT", fill="gray", font=fonts["small"])
    draw.text((col5_x, y_offset + 38), seat, fill=main_color, font=fonts["large"])

    # Row 2: Booking ref, Class
    draw.text((col2_x, y_offset + 100), "BOOKING REF", fill="gray", font=fonts["small"])
    draw.text((col2_x, y_offset + 118), booking_ref, fill="black", font=fonts["medium"])

    draw.text((col3_x, y_offset + 100), "CLASS", fill="gray", font=fonts["small"])
    draw.text((col3_x, y_offset + 118), travel_class, fill="black", font=fonts["medium"])

    draw.text((col4_x, y_offset + 100), "FLIGHT", fill="gray", font=fonts["small"])
    draw.text((col4_x, y_offset + 118), flight_number, fill="black", font=fonts["medium"])

    # Issue date and GST section (above barcode)
    info_y = y_offset + 170

    if issue_date:
        draw.text((30, info_y), "ISSUED", fill="gray", font=fonts["small"])
        draw.text((30, info_y + 18), issue_date.strftime("%d %b %Y"), fill="black", font=fonts["medium"])

    # GST breakdown
    subtotal = price - gst_amount
    draw.text((200, info_y), "SUBTOTAL (ex GST)", fill="gray", font=fonts["small"])
    draw.text((200, info_y + 18), f"${subtotal:.2f}", fill="black", font=fonts["medium"])

    draw.text((380, info_y), "GST (10%)", fill="gray", font=fonts["small"])
    draw.text((380, info_y + 18), f"${gst_amount:.2f}", fill="black", font=fonts["medium"])

    draw.text((520, info_y), "TOTAL", fill="gray", font=fonts["small"])
    draw.text((520, info_y + 18), f"AUD ${price:.2f}", fill=main_color, font=fonts["medium"])

    # Barcode area (simulated)
    barcode_y = height - 70
    draw.rectangle([(20, barcode_y), (width - 20, barcode_y + 50)], outline="lightgray")

    # Simulated barcode lines
    x = 30
    while x < width - 30:
        bar_width = random.choice([2, 3, 4])
        if random.random() > 0.3:
            draw.rectangle([(x, barcode_y + 10), (x + bar_width, barcode_y + 40)], fill="black")
        x += bar_width + random.randint(1, 3)

    # Save
    img.save(output_path, quality=95)


def draw_multi_leg_ticket(
    airline: dict,
    legs: list[dict],
    passenger: str,
    booking_ref: str,
    travel_class: str,
    total_price: float,
    output_path: Path,
    issue_date: datetime | None = None,
    gst_amount: float = 0.0,
) -> None:
    """
    Draw and save a multi-leg itinerary ticket image.

    Args:
        airline: Airline info dict
        legs: List of leg dicts, each containing:
              origin, destination, flight_date, flight_number, seat, gate
        passenger: Passenger name
        booking_ref: Booking reference
        travel_class: Travel class
        total_price: Total price for all legs
        output_path: Output file path
        issue_date: Date ticket was issued
        gst_amount: GST amount included in total
    """
    num_legs = len(legs)

    # Adjust height based on number of legs (increased footer for GST info)
    width = 900
    header_height = 70
    leg_height = 140
    footer_height = 150  # Increased to fit GST breakdown
    height = header_height + (leg_height * num_legs) + footer_height

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # Colors
    main_color = hex_to_rgb(airline["color"])
    accent_color = hex_to_rgb(airline["accent"])

    fonts = load_fonts()

    # Header bar with airline branding
    draw.rectangle([(0, 0), (width, header_height)], fill=main_color)
    draw.text((20, 18), airline["name"].upper(), fill="white", font=fonts["large"])
    draw.text((width - 180, 25), f"{num_legs}-LEG ITINERARY", fill=accent_color, font=fonts["medium"])

    # Itinerary label
    draw.rectangle([(width - 200, header_height), (width, header_height + 35)], fill=accent_color)
    draw.text((width - 180, header_height + 8), "E-TICKET", fill=main_color, font=fonts["small"])

    # Passenger and booking info (top section)
    draw.text((20, header_height + 10), "PASSENGER", fill="gray", font=fonts["small"])
    draw.text((20, header_height + 26), passenger, fill="black", font=fonts["medium"])

    draw.text((250, header_height + 10), "BOOKING REF", fill="gray", font=fonts["small"])
    draw.text((250, header_height + 26), booking_ref, fill="black", font=fonts["medium"])

    draw.text((420, header_height + 10), "CLASS", fill="gray", font=fonts["small"])
    draw.text((420, header_height + 26), travel_class, fill="black", font=fonts["medium"])

    # Draw separator line
    draw.line([(20, header_height + 55), (width - 20, header_height + 55)], fill="lightgray", width=1)

    # Draw each leg
    y_start = header_height + 65

    for i, leg in enumerate(legs):
        y = y_start + (i * leg_height)

        # Leg number indicator
        draw.rectangle([(20, y), (50, y + 25)], fill=main_color)
        draw.text((28, y + 4), str(i + 1), fill="white", font=fonts["medium"])

        # Origin
        draw.text((70, y), leg["origin"]["code"], fill=main_color, font=fonts["code_small"])
        draw.text((70, y + 25), leg["origin"]["city"], fill="gray", font=fonts["tiny"])

        # Arrow
        draw.text((150, y + 5), "→", fill=accent_color, font=fonts["medium"])

        # Destination
        draw.text((180, y), leg["destination"]["code"], fill=main_color, font=fonts["code_small"])
        draw.text((180, y + 25), leg["destination"]["city"], fill="gray", font=fonts["tiny"])

        # Flight details
        col1 = 300
        col2 = 420
        col3 = 540
        col4 = 660
        col5 = 780

        draw.text((col1, y), "DATE", fill="gray", font=fonts["tiny"])
        draw.text((col1, y + 14), leg["flight_date"].strftime("%d %b %Y"), fill="black", font=fonts["small"])

        draw.text((col2, y), "DEPART", fill="gray", font=fonts["tiny"])
        draw.text((col2, y + 14), leg["flight_date"].strftime("%H:%M"), fill="black", font=fonts["small"])

        draw.text((col3, y), "FLIGHT", fill="gray", font=fonts["tiny"])
        draw.text((col3, y + 14), leg["flight_number"], fill="black", font=fonts["small"])

        draw.text((col4, y), "GATE", fill="gray", font=fonts["tiny"])
        draw.text((col4, y + 14), leg["gate"], fill="black", font=fonts["small"])

        draw.text((col5, y), "SEAT", fill="gray", font=fonts["tiny"])
        draw.text((col5, y + 14), leg["seat"], fill=main_color, font=fonts["medium"])

        # Connection info (if not last leg)
        if i < num_legs - 1:
            next_leg = legs[i + 1]
            # Calculate layover
            arrival_est = leg["flight_date"] + timedelta(hours=random.uniform(1.5, 4))
            layover = next_leg["flight_date"] - arrival_est
            layover_hrs = layover.total_seconds() / 3600

            draw.text((70, y + 50), f"Connection at {leg['destination']['city']}", fill="gray", font=fonts["tiny"])
            draw.text((280, y + 50), f"Layover: {layover_hrs:.1f} hrs", fill="gray", font=fonts["tiny"])

            # Dashed separator
            for dash_x in range(20, width - 20, 10):
                draw.line([(dash_x, y + leg_height - 15), (dash_x + 5, y + leg_height - 15)], fill="lightgray", width=1)

    # Issue date and GST breakdown section
    info_y = height - footer_height + 10

    if issue_date:
        draw.text((20, info_y), "ISSUED", fill="gray", font=fonts["small"])
        draw.text((20, info_y + 18), issue_date.strftime("%d %b %Y"), fill="black", font=fonts["medium"])

    # GST breakdown
    subtotal = total_price - gst_amount
    draw.text((180, info_y), "SUBTOTAL (ex GST)", fill="gray", font=fonts["small"])
    draw.text((180, info_y + 18), f"${subtotal:.2f}", fill="black", font=fonts["medium"])

    draw.text((360, info_y), "GST (10%)", fill="gray", font=fonts["small"])
    draw.text((360, info_y + 18), f"${gst_amount:.2f}", fill="black", font=fonts["medium"])

    draw.text((500, info_y), "TOTAL", fill="gray", font=fonts["small"])
    draw.text((500, info_y + 18), f"AUD ${total_price:.2f}", fill=main_color, font=fonts["medium"])

    # Footer with barcode
    barcode_y = info_y + 55
    draw.rectangle([(20, barcode_y), (width - 20, barcode_y + 50)], outline="lightgray")

    # Simulated barcode
    x = 30
    while x < width - 30:
        bar_width = random.choice([2, 3, 4])
        if random.random() > 0.3:
            draw.rectangle([(x, barcode_y + 10), (x + bar_width, barcode_y + 40)], fill="black")
        x += bar_width + random.randint(1, 3)

    # Save
    img.save(output_path, quality=95)


def generate_ticket_batch(
    output_dir: Path,
    num_tickets: int = 20,
    start_date: datetime | None = None,
    date_range_days: int = 90,
    seed: int | None = None,
) -> list[dict]:
    """
    Generate a batch of synthetic single-leg airline tickets.

    Args:
        output_dir: Directory to save ticket images
        num_tickets: Number of tickets to generate
        start_date: Earliest travel date (defaults to today)
        date_range_days: Range of days for travel dates
        seed: Random seed for reproducibility

    Returns:
        List of ticket metadata dictionaries
    """
    if seed is not None:
        random.seed(seed)

    if start_date is None:
        start_date = datetime.now()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tickets = []

    for i in range(num_tickets):
        # Select random elements
        airline = random.choice(AIRLINES)
        origin, destination = random.sample(CITIES, 2)  # Ensure different cities
        passenger = generate_passenger_name()
        travel_class = random.choices(CLASSES, weights=CLASS_WEIGHTS, k=1)[0]

        # Generate flight details
        flight_date = start_date + timedelta(
            days=random.randint(0, date_range_days),
            hours=random.randint(5, 22),
            minutes=random.choice([0, 15, 30, 45]),
        )
        flight_number = generate_flight_number(airline["code"])
        booking_ref = generate_booking_ref()
        seat = generate_seat()
        gate = generate_gate()
        price = generate_price(travel_class)

        # Generate issue date (1-30 days before travel date)
        issue_date = flight_date - timedelta(days=random.randint(1, 30))

        # Calculate GST (domestic flights include 10% GST)
        gst_amount, _subtotal = calculate_gst(price, gst_free=False)

        # Generate filename
        filename = f"ticket_{i+1:03d}_{airline['code']}_{origin['code']}_{destination['code']}.png"
        output_path = output_dir / filename

        # Draw ticket
        draw_ticket(
            airline=airline,
            origin=origin,
            destination=destination,
            passenger=passenger,
            flight_date=flight_date,
            flight_number=flight_number,
            booking_ref=booking_ref,
            seat=seat,
            gate=gate,
            travel_class=travel_class,
            price=price,
            output_path=output_path,
            issue_date=issue_date,
            gst_amount=gst_amount,
        )

        # Store metadata for ground truth - expense claim format with GST
        ticket_meta = {
            "filename": filename,
            # Expense claim essentials: WHO, WHERE, WHEN, HOW, HOW MUCH, PROVIDER
            "PASSENGER_NAME": passenger,
            "TRAVEL_MODE": "plane",
            "TRAVEL_ROUTE": f"{origin['city']} → {destination['city']}",
            "TRAVEL_DATES": flight_date.strftime("%d %b %Y"),
            "INVOICE_DATE": issue_date.strftime("%d %b %Y"),
            "GST_AMOUNT": f"${gst_amount:.2f}",
            "TOTAL_AMOUNT": f"${price:.2f}",
            "SUPPLIER_NAME": airline["name"],
        }
        tickets.append(ticket_meta)

        print(f"Generated: {filename}")

    return tickets


def generate_multi_leg_batch(
    output_dir: Path,
    num_tickets: int = 20,
    start_date: datetime | None = None,
    date_range_days: int = 90,
    seed: int | None = None,
    min_legs: int = 2,
    max_legs: int = 4,
    round_trip_probability: float = 0.4,
) -> list[dict]:
    """
    Generate a batch of synthetic multi-leg airline tickets.

    Args:
        output_dir: Directory to save ticket images
        num_tickets: Number of tickets to generate
        start_date: Earliest travel date (defaults to today)
        date_range_days: Range of days for travel dates
        seed: Random seed for reproducibility
        min_legs: Minimum legs per itinerary (default 2)
        max_legs: Maximum legs per itinerary (default 4)
        round_trip_probability: Probability of generating a round trip (default 0.4)

    Returns:
        List of ticket metadata dictionaries (one per ticket, with leg details nested)
    """
    if seed is not None:
        random.seed(seed)

    if start_date is None:
        start_date = datetime.now()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tickets = []

    for i in range(num_tickets):
        # Select random elements
        airline = random.choice(AIRLINES)
        passenger = generate_passenger_name()
        travel_class = random.choices(CLASSES, weights=CLASS_WEIGHTS, k=1)[0]
        booking_ref = generate_booking_ref()

        # Determine number of legs
        num_legs = random.randint(min_legs, max_legs)

        # Decide if this is a round trip (returns to origin)
        is_round_trip = random.random() < round_trip_probability

        # Select cities for the route (ensuring no immediate repeats)
        route_cities = []
        available = CITIES.copy()

        if is_round_trip:
            # For round trips: pick origin, then intermediate stops, then return to origin
            origin = random.choice(available)
            route_cities.append(origin)

            # Track all visited cities to avoid revisiting until return
            visited = {origin["code"]}

            # Add intermediate cities (num_legs - 1 of them, since last leg returns home)
            for _ in range(num_legs - 1):
                available = [c for c in CITIES if c["code"] not in visited]
                if not available:
                    # If we've visited all cities, allow revisits except immediate repeat
                    available = [c for c in CITIES if c != route_cities[-1]]
                city = random.choice(available)
                route_cities.append(city)
                visited.add(city["code"])

            # Final destination is the origin (round trip)
            route_cities.append(origin)
        else:
            # One-way multi-city: all different cities
            for _ in range(num_legs + 1):  # Need n+1 cities for n legs
                city = random.choice(available)
                route_cities.append(city)
                # Remove this city from available to avoid immediate return
                available = [c for c in CITIES if c != city]

        # Generate leg details
        legs = []
        current_date = start_date + timedelta(days=random.randint(0, date_range_days))

        for leg_idx in range(num_legs):
            origin = route_cities[leg_idx]
            destination = route_cities[leg_idx + 1]

            # Set departure time
            flight_date = current_date + timedelta(
                hours=random.randint(6, 20),
                minutes=random.choice([0, 15, 30, 45]),
            )

            leg = {
                "origin": origin,
                "destination": destination,
                "flight_date": flight_date,
                "flight_number": generate_flight_number(airline["code"]),
                "seat": generate_seat(),
                "gate": generate_gate(),
            }
            legs.append(leg)

            # Next leg departs after layover (1-8 hours, or next day)
            if random.random() < 0.3:  # 30% chance of overnight
                current_date = flight_date + timedelta(days=1)
            else:
                current_date = flight_date + timedelta(hours=random.uniform(2, 6))

        # Calculate total price
        base_price = generate_price(travel_class)
        total_price = round(base_price * num_legs * random.uniform(0.8, 1.1), 2)

        # Generate issue date (1-30 days before first travel date)
        first_travel_date = legs[0]["flight_date"]
        issue_date = first_travel_date - timedelta(days=random.randint(1, 30))

        # Calculate GST (domestic flights include 10% GST)
        gst_amount, _subtotal = calculate_gst(total_price, gst_free=False)

        # Generate filename (origin to final destination)
        first_origin = route_cities[0]["code"]
        final_dest = route_cities[-1]["code"]
        filename = f"itinerary_{i+1:03d}_{airline['code']}_{first_origin}_{final_dest}_{num_legs}leg.png"
        output_path = output_dir / filename

        # Draw multi-leg ticket
        draw_multi_leg_ticket(
            airline=airline,
            legs=legs,
            passenger=passenger,
            booking_ref=booking_ref,
            travel_class=travel_class,
            total_price=total_price,
            output_path=output_path,
            issue_date=issue_date,
            gst_amount=gst_amount,
        )

        # Store metadata for ground truth - expense claim format with GST

        # Build route with → separator: "Sydney → Melbourne → Brisbane → Sydney"
        route_parts = [route_cities[0]["city"]]  # Start with origin
        for leg in legs:
            route_parts.append(leg["destination"]["city"])
        travel_route = " → ".join(route_parts)

        # Dates with " | " separator (with year for itineraries now)
        travel_dates = " | ".join(leg["flight_date"].strftime("%d %b %Y") for leg in legs)

        ticket_meta = {
            "filename": filename,
            # Expense claim essentials: WHO, WHERE, WHEN, HOW, HOW MUCH, PROVIDER, GST
            "PASSENGER_NAME": passenger,
            "TRAVEL_MODE": "plane",
            "TRAVEL_ROUTE": travel_route,
            "TRAVEL_DATES": travel_dates,
            "INVOICE_DATE": issue_date.strftime("%d %b %Y"),
            "GST_AMOUNT": f"${gst_amount:.2f}",
            "TOTAL_AMOUNT": f"${total_price:.2f}",
            "SUPPLIER_NAME": airline["name"],
        }

        tickets.append(ticket_meta)
        print(f"Generated: {filename} ({num_legs} legs)")

    return tickets


def save_ground_truth(tickets: list[dict], output_path: Path) -> None:
    """Save ticket metadata as CSV ground truth file."""
    import csv

    output_path = Path(output_path)

    if not tickets:
        return

    # Collect all unique field names across all tickets (handles variable leg counts)
    all_fields = []
    seen = set()
    for ticket in tickets:
        for key in ticket.keys():
            if key not in seen:
                all_fields.append(key)
                seen.add(key)

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(tickets)

    print(f"Ground truth saved: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic airline ticket images")
    parser.add_argument(
        "-n", "--num-tickets", type=int, default=20, help="Number of tickets to generate (default: 20)"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="data/synthetic_tickets",
        help="Output directory (default: data/synthetic_tickets)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument(
        "--multi-leg",
        action="store_true",
        help="Generate multi-leg itineraries instead of single tickets",
    )
    parser.add_argument(
        "--min-legs",
        type=int,
        default=2,
        help="Minimum legs per itinerary for multi-leg mode (default: 2)",
    )
    parser.add_argument(
        "--max-legs",
        type=int,
        default=4,
        help="Maximum legs per itinerary for multi-leg mode (default: 4)",
    )
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="Generate a mix of single-leg and multi-leg tickets (50/50 split)",
    )
    parser.add_argument(
        "--single-count",
        type=int,
        default=None,
        help="Exact number of single-leg tickets (use with --multi-count)",
    )
    parser.add_argument(
        "--multi-count",
        type=int,
        default=None,
        help="Exact number of multi-leg tickets (use with --single-count)",
    )
    parser.add_argument(
        "--round-trip-prob",
        type=float,
        default=0.4,
        help="Probability of multi-leg being round trip (default: 0.4)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Clear output directory first
    if output_dir.exists():
        for f in output_dir.glob("*.png"):
            f.unlink()
        for f in output_dir.glob("*.csv"):
            f.unlink()

    if args.single_count is not None and args.multi_count is not None:
        # Explicit counts specified
        single_tickets = generate_ticket_batch(
            output_dir=output_dir,
            num_tickets=args.single_count,
            seed=args.seed,
        )

        multi_tickets = generate_multi_leg_batch(
            output_dir=output_dir,
            num_tickets=args.multi_count,
            seed=args.seed + 1000 if args.seed else None,
            min_legs=args.min_legs,
            max_legs=args.max_legs,
            round_trip_probability=args.round_trip_prob,
        )

        tickets = single_tickets + multi_tickets
        print(f"\nGenerated {args.single_count} single-leg + {args.multi_count} multi-leg tickets")

    elif args.mixed:
        # Generate half single, half multi-leg
        single_count = args.num_tickets // 2
        multi_count = args.num_tickets - single_count

        single_tickets = generate_ticket_batch(
            output_dir=output_dir,
            num_tickets=single_count,
            seed=args.seed,
        )

        multi_tickets = generate_multi_leg_batch(
            output_dir=output_dir,
            num_tickets=multi_count,
            seed=args.seed + 1000 if args.seed else None,
            min_legs=args.min_legs,
            max_legs=args.max_legs,
            round_trip_probability=args.round_trip_prob,
        )

        tickets = single_tickets + multi_tickets
        print(f"\nGenerated {single_count} single-leg + {multi_count} multi-leg tickets")

    elif args.multi_leg:
        tickets = generate_multi_leg_batch(
            output_dir=output_dir,
            num_tickets=args.num_tickets,
            seed=args.seed,
            min_legs=args.min_legs,
            max_legs=args.max_legs,
            round_trip_probability=args.round_trip_prob,
        )
        print(f"\nGenerated {len(tickets)} multi-leg itineraries")

    else:
        tickets = generate_ticket_batch(
            output_dir=output_dir,
            num_tickets=args.num_tickets,
            seed=args.seed,
        )
        print(f"\nGenerated {len(tickets)} single-leg tickets")

    # Save ground truth CSV
    ground_truth_path = output_dir / "ground_truth.csv"
    save_ground_truth(tickets, ground_truth_path)

    print(f"Output directory: {output_dir}")
    print(f"Ground truth: {ground_truth_path}")
