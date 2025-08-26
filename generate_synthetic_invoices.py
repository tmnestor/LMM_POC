#!/usr/bin/env python3
"""
Synthetic ATO Compliant Tax Invoice Generator

Generates realistic Australian tax invoices that comply with ATO requirements:
- Sales under $1,000: Seller identity, ABN, date, items, GST
- Sales $1,000+: Also requires buyer identity or ABN
- Proper GST calculations (1/11 of total when included)
"""

import csv
import json
import random
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Dict, List

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL not available - only JSON/HTML generation supported")

class ATOTaxInvoiceGenerator:
    def __init__(self):
        """Initialize with realistic Australian business data."""
        
        # Field schema mapping for ground truth CSV generation (matches field_schema_v2.yaml)
        self.field_schema_mapping = {
            # Document metadata
            "DOCUMENT_TYPE": "document_type",
            "INVOICE_NUMBER": "invoice_number", 
            "INVOICE_DATE": "invoice_date",
            "DUE_DATE": "due_date",
            
            # Seller information (ATO mandatory)
            "SUPPLIER_NAME": "supplier_name",
            "BUSINESS_ABN": "business_abn",
            "BUSINESS_ADDRESS": "business_address", 
            "BUSINESS_PHONE": "business_phone",
            "SUPPLIER_WEBSITE": None,  # Not included in this generator
            
            # Customer information (conditional for $1000+)
            "PAYER_NAME": "payer_name",
            "PAYER_ABN": "payer_abn", 
            "PAYER_ADDRESS": "payer_address",
            "PAYER_PHONE": "payer_phone",
            "PAYER_EMAIL": None,  # Not included in this generator
            
            # Line items (ATO mandatory) 
            "LINE_ITEM_DESCRIPTIONS": "line_item_descriptions",
            "LINE_ITEM_QUANTITIES": "line_item_quantities",
            "LINE_ITEM_PRICES": "line_item_prices",
            
            # Financial amounts (ATO mandatory)
            "SUBTOTAL_AMOUNT": "subtotal_amount",
            "GST_AMOUNT": "gst_amount", 
            "TOTAL_AMOUNT": "total_amount",
            
            # Fields not applicable to invoices (bank statement fields)
            "BANK_NAME": None,
            "BANK_BSB_NUMBER": None,
            "BANK_ACCOUNT_NUMBER": None,
            "BANK_ACCOUNT_HOLDER": None,
            "STATEMENT_DATE_RANGE": None,
            "ACCOUNT_OPENING_BALANCE": None,
            "ACCOUNT_CLOSING_BALANCE": None,
            "TOTAL_CREDITS": None,
            "TOTAL_DEBITS": None
        }
        
        # Australian business names
        self.business_names = [
            "Aussie Office Supplies Pty Ltd",
            "Sydney Tech Solutions",
            "Melbourne Hardware Store",
            "Brisbane Catering Services",
            "Perth Building Materials",
            "Adelaide Medical Equipment",
            "Canberra Consulting Group",
            "Gold Coast Retail Supplies",
            "Hobart Fresh Produce",
            "Darwin Industrial Services"
        ]
        
        # Australian addresses
        self.business_addresses = [
            "123 Collins Street, Melbourne VIC 3000",
            "456 George Street, Sydney NSW 2000",
            "789 Queen Street, Brisbane QLD 4000",
            "321 Murray Street, Perth WA 6000",
            "654 King William Street, Adelaide SA 5000",
            "987 London Circuit, Canberra ACT 2600",
            "147 Surfers Paradise Blvd, Gold Coast QLD 4217",
            "258 Elizabeth Street, Hobart TAS 7000",
            "369 Smith Street, Darwin NT 0800",
            "741 Flinders Street, Adelaide SA 5000"
        ]
        
        # Customer names
        self.customer_names = [
            "John Smith",
            "Sarah Johnson", 
            "Michael Chen",
            "Emma Williams",
            "David Brown",
            "Lisa Wilson",
            "James Taylor",
            "Sophie Martin",
            "Robert Davis",
            "Anna Thompson"
        ]
        
        # Customer addresses
        self.customer_addresses = [
            "45 Residential Ave, Suburb NSW 2100",
            "78 Home Street, Township QLD 4000",
            "12 Living Lane, District VIC 3150",
            "34 Family Road, Area WA 6050",
            "56 House Court, Region SA 5100",
            "89 Apartment Blvd, Zone ACT 2900",
            "23 Unit Street, Locale TAS 7200",
            "67 Flat Avenue, Precinct NT 0850"
        ]
        
        # Products and services
        self.products = [
            ("Office Chair", 89.95, "Ergonomic office chair"),
            ("Laptop Computer", 1299.00, "15-inch business laptop"),
            ("Printer Paper", 12.50, "A4 copy paper - 500 sheets"),
            ("Wireless Mouse", 34.95, "Bluetooth wireless mouse"),
            ("Monitor Stand", 67.00, "Adjustable monitor stand"),
            ("Desk Lamp", 45.50, "LED desk lamp"),
            ("Filing Cabinet", 189.00, "4-drawer steel filing cabinet"),
            ("Whiteboard", 125.00, "Magnetic whiteboard 120x90cm"),
            ("Shredder", 234.00, "Cross-cut document shredder"),
            ("Coffee Machine", 399.00, "Commercial coffee machine"),
            ("Cleaning Service", 150.00, "Office cleaning service"),
            ("IT Support", 95.00, "Technical support - per hour"),
            ("Catering Package", 25.00, "Lunch catering per person"),
            ("Training Workshop", 350.00, "Professional development training"),
            ("Consultation", 180.00, "Business consultation - per hour")
        ]
        
    def generate_abn(self) -> str:
        """Generate a valid-format ABN (11 digits)."""
        return f"{random.randint(10, 99)} {random.randint(100, 999)} {random.randint(100, 999)} {random.randint(100, 999)}"
    
    def generate_phone(self) -> str:
        """Generate Australian phone number."""
        area_codes = ["02", "03", "07", "08"]
        area = random.choice(area_codes)
        number = f"{random.randint(1000, 9999)} {random.randint(1000, 9999)}"
        return f"({area}) {number}"
    
    def calculate_gst(self, amount: Decimal) -> Decimal:
        """Calculate GST (1/11 of total when GST-inclusive)."""
        gst = amount / Decimal("11")
        return gst.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    
    def calculate_ex_gst(self, amount: Decimal) -> Decimal:
        """Calculate amount excluding GST."""
        return amount - self.calculate_gst(amount)
    
    def generate_invoice_data(self, target_amount: str = "random") -> Dict:
        """
        Generate complete invoice data.
        
        Args:
            target_amount: "under_1000", "over_1000", or "random"
        """
        
        # Basic invoice details
        invoice_num = f"INV-{random.randint(10000, 99999)}"
        invoice_date = datetime.now() - timedelta(days=random.randint(0, 90))
        due_date = invoice_date + timedelta(days=random.randint(7, 30))
        
        # Seller details (always required)
        business_idx = random.randint(0, len(self.business_names) - 1)
        seller_name = self.business_names[business_idx]
        seller_address = self.business_addresses[business_idx]
        seller_abn = self.generate_abn()
        seller_phone = self.generate_phone()
        
        # Generate line items based on target amount
        if target_amount == "under_1000":
            target_total = random.uniform(50, 999)
        elif target_amount == "over_1000":
            target_total = random.uniform(1000, 5000)
        else:  # random
            target_total = random.uniform(50, 3000)
        
        line_items = self._generate_line_items(target_total)
        
        # Calculate totals
        subtotal = sum(Decimal(str(item["total"])) for item in line_items)
        gst_amount = self.calculate_gst(subtotal)
        total_amount = subtotal
        subtotal_ex_gst = self.calculate_ex_gst(subtotal)
        
        # Customer details (required for $1000+)
        needs_customer = total_amount >= 1000
        customer_name = None
        customer_abn = None
        customer_address = None
        customer_phone = None
        
        if needs_customer:
            # 70% chance of name, 30% chance of ABN for business customers
            if random.random() < 0.7:
                customer_name = random.choice(self.customer_names)
                customer_address = random.choice(self.customer_addresses)
                customer_phone = self.generate_phone()
            else:
                customer_name = random.choice(self.business_names)
                customer_abn = self.generate_abn()
                customer_address = random.choice(self.business_addresses)
                customer_phone = self.generate_phone()
        
        return {
            # ATO Mandatory fields
            "document_type": "TAX INVOICE",
            "invoice_number": invoice_num,
            "invoice_date": invoice_date.strftime("%d/%m/%Y"),
            "due_date": due_date.strftime("%d/%m/%Y"),
            
            # Seller details (mandatory)
            "supplier_name": seller_name,
            "business_abn": seller_abn,
            "business_address": seller_address,
            "business_phone": seller_phone,
            
            # Customer details (conditional)
            "payer_name": customer_name,
            "payer_abn": customer_abn,
            "payer_address": customer_address,
            "payer_phone": customer_phone,
            
            # Line items (mandatory)
            "line_items": line_items,
            "line_item_descriptions": [item["description"] for item in line_items],
            "line_item_quantities": [str(item["quantity"]) for item in line_items],
            "line_item_prices": [f"${item['unit_price']:.2f}" for item in line_items],
            
            # Financial totals (mandatory)
            "subtotal_amount": f"${subtotal_ex_gst:.2f}",
            "gst_amount": f"${gst_amount:.2f}",
            "total_amount": f"${total_amount:.2f}",
            
            # Additional details
            "gst_statement": "Total price includes GST" if gst_amount > 0 else None,
            "payment_terms": f"Payment due within {(due_date - invoice_date).days} days",
            
            # Metadata
            "ato_compliant": True,
            "requires_buyer_details": needs_customer,
            "generation_timestamp": datetime.now().isoformat()
        }
    
    def _generate_line_items(self, target_total: float) -> List[Dict]:
        """Generate realistic line items that sum to approximately target_total."""
        items = []
        remaining = target_total
        
        # Generate 1-5 line items
        num_items = random.randint(1, min(5, len(self.products)))
        
        for i in range(num_items):
            if i == num_items - 1:  # Last item - use remaining amount
                unit_price = remaining
                if unit_price < 5:
                    unit_price = random.uniform(5, 50)
                quantity = 1
            else:
                # Pick random product and adjust price
                product_name, base_price, description = random.choice(self.products)
                
                # Vary the price ±30%
                price_variation = random.uniform(0.7, 1.3)
                unit_price = base_price * price_variation
                
                # Ensure we don't exceed remaining budget
                max_price = remaining * 0.8  # Leave room for other items
                unit_price = min(unit_price, max_price)
                
                quantity = random.randint(1, 3)
                
                # Adjust if total exceeds remaining
                line_total = unit_price * quantity
                if line_total > remaining * 0.8:
                    quantity = 1
                    unit_price = min(unit_price, remaining * 0.8)
            
            line_total = unit_price * quantity
            remaining -= line_total
            
            # Get product details
            product_name, _, description = random.choice(self.products)
            
            items.append({
                "description": product_name,
                "quantity": quantity,
                "unit_price": unit_price,
                "total": line_total
            })
            
            if remaining <= 0:
                break
        
        return items
    
    
    def generate_invoice_png(self, invoice_data: Dict, output_path: str) -> None:
        """Generate a PNG image of the tax invoice."""
        
        if not PIL_AVAILABLE:
            print("⚠️ Skipping PNG generation - PIL not available")
            return
            
        # Image dimensions
        width, height = 800, 1000
        background_color = "white"
        text_color = "black"
        
        # Create image and drawing context
        img = Image.new('RGB', (width, height), background_color)
        draw = ImageDraw.Draw(img)
        
        # Try to load fonts, fall back to default if not available
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            header_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            normal_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
            small_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except (OSError, IOError):
            # Fallback to default font
            title_font = ImageFont.load_default()
            header_font = ImageFont.load_default()
            normal_font = ImageFont.load_default()
            small_font = ImageFont.load_default()
        
        y_pos = 30
        
        # Title - TAX INVOICE
        title_text = "TAX INVOICE"
        title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        draw.text(((width - title_width) // 2, y_pos), title_text, fill=text_color, font=title_font)
        y_pos += 50
        
        # Business Information (Centered)
        business_info = [
            invoice_data['supplier_name'],
            f"ABN: {invoice_data['business_abn']}", 
            invoice_data['business_address'],
            f"Ph: {invoice_data['business_phone']}"
        ]
        
        for line in business_info:
            bbox = draw.textbbox((0, 0), line, font=header_font)
            line_width = bbox[2] - bbox[0]
            draw.text(((width - line_width) // 2, y_pos), line, fill=text_color, font=header_font)
            y_pos += 25
        
        y_pos += 20
        
        # Draw separator line
        draw.line([(50, y_pos), (width-50, y_pos)], fill=text_color, width=2)
        y_pos += 30
        
        # Invoice Details (Left side)
        invoice_details = [
            f"Invoice Number: {invoice_data['invoice_number']}",
            f"Invoice Date: {invoice_data['invoice_date']}",
            f"Due Date: {invoice_data['due_date']}"
        ]
        
        for detail in invoice_details:
            draw.text((50, y_pos), detail, fill=text_color, font=normal_font)
            y_pos += 22
        
        y_pos += 20
        
        # Customer Information (if present for $1000+ invoices)
        if invoice_data.get('payer_name') or invoice_data.get('payer_abn'):
            draw.text((50, y_pos), "BILL TO:", fill=text_color, font=header_font)
            y_pos += 25
            
            customer_info = []
            if invoice_data.get('payer_name'):
                customer_info.append(invoice_data['payer_name'])
            if invoice_data.get('payer_abn'):
                customer_info.append(f"ABN: {invoice_data['payer_abn']}")
            if invoice_data.get('payer_address'):
                customer_info.append(invoice_data['payer_address'])
            if invoice_data.get('payer_phone'):
                customer_info.append(f"Ph: {invoice_data['payer_phone']}")
            
            for info in customer_info:
                draw.text((70, y_pos), info, fill=text_color, font=small_font)
                y_pos += 18
            
            y_pos += 20
        
        # Line Items Table Header
        draw.text((50, y_pos), "DESCRIPTION", fill=text_color, font=header_font)
        draw.text((400, y_pos), "QTY", fill=text_color, font=header_font)
        draw.text((500, y_pos), "PRICE", fill=text_color, font=header_font)
        draw.text((650, y_pos), "TOTAL", fill=text_color, font=header_font)
        y_pos += 25
        
        # Draw table header line
        draw.line([(50, y_pos), (width-50, y_pos)], fill=text_color, width=1)
        y_pos += 15
        
        # Line Items
        for item in invoice_data['line_items']:
            # Truncate long descriptions
            desc = item['description'][:35] + "..." if len(item['description']) > 35 else item['description']
            
            draw.text((50, y_pos), desc, fill=text_color, font=small_font)
            draw.text((400, y_pos), str(item['quantity']), fill=text_color, font=small_font)
            draw.text((500, y_pos), f"${item['unit_price']:.2f}", fill=text_color, font=small_font)
            draw.text((650, y_pos), f"${item['total']:.2f}", fill=text_color, font=small_font)
            y_pos += 20
        
        y_pos += 20
        
        # Draw separator line
        draw.line([(400, y_pos), (width-50, y_pos)], fill=text_color, width=1)
        y_pos += 15
        
        # Financial Totals (Right aligned)
        totals = [
            ("Subtotal (ex GST):", invoice_data['subtotal_amount']),
            ("GST:", invoice_data['gst_amount']),
            ("TOTAL:", invoice_data['total_amount'])
        ]
        
        for label, amount in totals:
            draw.text((500, y_pos), label, fill=text_color, font=normal_font)
            draw.text((650, y_pos), amount, fill=text_color, font=normal_font)
            y_pos += 22
        
        y_pos += 30
        
        # GST Statement
        if invoice_data.get('gst_statement'):
            draw.text((50, y_pos), invoice_data['gst_statement'], fill=text_color, font=small_font)
            y_pos += 20
        
        # Payment terms
        draw.text((50, y_pos), invoice_data['payment_terms'], fill=text_color, font=small_font)
        y_pos += 30
        
        # ATO Compliance notice
        ato_notice = "This tax invoice contains all information required under Australian tax legislation."
        draw.text((50, y_pos), ato_notice, fill=text_color, font=small_font)
        
        # Save the image
        img.save(output_path, 'PNG', quality=95, dpi=(300, 300))
        print(f"📄 Generated PNG: {Path(output_path).name}")
    
    def map_to_extraction_fields(self, invoice_data: Dict, image_filename: str) -> Dict[str, str]:
        """Map invoice data to extraction field format for ground truth CSV."""
        
        result = {"image_file": image_filename}
        
        # Map each field from schema to invoice data
        for csv_field, data_key in self.field_schema_mapping.items():
            if data_key is None:
                # Field not applicable to invoices
                result[csv_field] = "NOT_FOUND"
            elif data_key in invoice_data and invoice_data[data_key] is not None:
                value = invoice_data[data_key]
                
                # Handle list fields (pipe-separated format)
                if isinstance(value, list):
                    result[csv_field] = " | ".join(str(v) for v in value)
                else:
                    result[csv_field] = str(value)
            else:
                result[csv_field] = "NOT_FOUND"
        
        return result
    
    def generate_ground_truth_csv(self, invoices_data: List[Dict], csv_path: str, append_mode: bool = False) -> None:
        """Generate or append to ground truth CSV file for model evaluation."""
        
        # Define CSV headers (all extraction fields)
        fieldnames = ["image_file"] + list(self.field_schema_mapping.keys())
        
        csv_file = Path(csv_path)
        
        # Determine write mode and whether to write header
        write_header = True
        mode = 'w'
        
        if append_mode and csv_file.exists():
            mode = 'a'
            write_header = False
            print(f"📝 Appending to existing CSV: {csv_file}")
        else:
            print(f"📝 Creating new CSV: {csv_file}")
        
        with csv_file.open(mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if write_header:
                writer.writeheader()
            
            for invoice_data in invoices_data:
                image_filename = invoice_data.get('image_filename', f"invoice_{invoice_data.get('invoice_number', 'unknown')}.png")
                row_data = self.map_to_extraction_fields(invoice_data, image_filename)
                writer.writerow(row_data)
        
        action = "Appended to" if mode == 'a' else "Generated"
        print(f"✅ {action} ground truth CSV: {csv_file}")
        print(f"📊 Fields mapped: {len(fieldnames)} columns, {len(invoices_data)} rows")
    
    def generate_batch(self, count: int = 10, output_dir: str = "synthetic_invoices", with_ground_truth: bool = True, 
                      start_number: int = 1, append_csv: bool = False) -> List[Dict]:
        """Generate a batch of synthetic invoices with optional ground truth CSV.
        
        Args:
            count: Number of invoices to generate
            output_dir: Directory to save files
            with_ground_truth: Whether to generate ground truth CSV
            start_number: Starting invoice number (e.g., 21 to generate from 021)
            append_csv: Whether to append to existing CSV instead of overwriting
        """
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        invoices = []
        
        # Generate mix of under/over $1000 invoices
        for i in range(count):
            if i < count // 3:
                target = "under_1000"
            elif i < 2 * count // 3:
                target = "over_1000"
            else:
                target = "random"
            
            invoice_data = self.generate_invoice_data(target)
            
            # Add image filename for ground truth tracking - using start_number
            invoice_num = start_number + i
            image_filename = f"synthetic_invoice_{invoice_num:03d}.png"
            invoice_data['image_filename'] = image_filename
            
            invoices.append(invoice_data)
            
            # Save as PNG image (primary format)
            png_file = output_path / image_filename
            self.generate_invoice_png(invoice_data, str(png_file))
            
            print(f"Generated invoice {invoice_num}: {invoice_data['invoice_number']} - {invoice_data['total_amount']} "
                  f"({'Requires buyer details' if invoice_data['requires_buyer_details'] else 'No buyer details required'})")
        
        # Generate ground truth CSV if requested
        if with_ground_truth:
            csv_path = output_path / "ground_truth.csv"
            self.generate_ground_truth_csv(invoices, str(csv_path), append_mode=append_csv)
        
        # Generate summary
        summary = {
            "total_generated": count,
            "under_1000": sum(1 for inv in invoices if not inv['requires_buyer_details']),
            "over_1000": sum(1 for inv in invoices if inv['requires_buyer_details']),
            "generation_date": datetime.now().isoformat(),
            "ato_compliant": True,
            "ground_truth_csv": with_ground_truth
        }
        
        summary_file = output_path / "generation_summary.json"
        with summary_file.open('w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Generated {count} ATO compliant tax invoices in {output_path}")
        print(f"   Under $1,000: {summary['under_1000']}")
        print(f"   $1,000 or more: {summary['over_1000']}")
        if with_ground_truth:
            print("   Ground truth CSV: ✅ Generated")
        
        return invoices


def main():
    """Generate synthetic tax invoices with ground truth CSV."""
    
    generator = ATOTaxInvoiceGenerator()
    
    # Generate 20 sample invoices as PNG files with ground truth CSV
    invoices = generator.generate_batch(
        count=20, 
        output_dir="ato_synthetic_invoices", 
        with_ground_truth=True
    )
    
    print("\n📊 Sample Invoice Data:")
    print(f"   Total invoices: {len(invoices)}")
    print(f"   Average amount: ${sum(float(inv['total_amount'][1:]) for inv in invoices) / len(invoices):.2f}")
    print(f"   ATO compliant: {all(inv['ato_compliant'] for inv in invoices)}")


if __name__ == "__main__":
    main()