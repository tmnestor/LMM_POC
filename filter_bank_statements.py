import shutil
from pathlib import Path

import pandas as pd

# Load ground truth data
ground_truth = pd.read_csv('evaluation_data/ground_truth.csv')

# Filter for BANK_STATEMENT in DOCUMENT_TYPE
bank_statements = ground_truth[ground_truth['DOCUMENT_TYPE'].str.contains('BANK_STATEMENT', na=False)]

# Get image_name values
image_names = bank_statements['image_name'].tolist()

print('Bank statement image names:')
for img in image_names:
    print(f'  {img}')

print(f'\nTotal: {len(image_names)} bank statement images')

# Save filtered list to CSV
bank_statements[['image_name', 'DOCUMENT_TYPE']].to_csv('output/bank_statement_images.csv', index=False)
print('\nSaved list to: output/bank_statement_images.csv')

# Create directory for bank statement images
output_dir = Path('evaluation_data/bank_statements')
output_dir.mkdir(parents=True, exist_ok=True)

# Copy bank statement images to new directory
source_dir = Path('evaluation_data/images')
copied_count = 0
missing_count = 0

print(f'\nCopying images to: {output_dir}')
for img_name in image_names:
    source_path = source_dir / img_name
    if source_path.exists():
        dest_path = output_dir / img_name
        shutil.copy2(source_path, dest_path)
        copied_count += 1
        print(f'  ✓ Copied: {img_name}')
    else:
        missing_count += 1
        print(f'  ✗ Missing: {img_name}')

print('\nSummary:')
print(f'  Copied: {copied_count} images')
print(f'  Missing: {missing_count} images')
print(f'  Destination: {output_dir.absolute()}')
