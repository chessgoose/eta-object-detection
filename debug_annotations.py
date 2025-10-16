"""
Debug script to inspect your annotations
"""

import json

ann_file = "/home/lyz6/AMROD/datasets/cityscapes/annotations/instancesonly_filtered_gtFine_val.json"

with open(ann_file, 'r') as f:
    coco_data = json.load(f)

print(f"Total images: {len(coco_data['images'])}")
print(f"Total annotations: {len(coco_data['annotations'])}")

# Check category IDs
category_ids = set()
for ann in coco_data['annotations']:
    category_ids.add(ann['category_id'])

print(f"\nUnique category IDs in annotations: {sorted(category_ids)}")

# Check categories
if 'categories' in coco_data:
    print("\nCategories:")
    for cat in coco_data['categories']:
        print(f"  ID {cat['id']}: {cat['name']}")

# Check image filenames
print("\nSample image filenames:")
for img in coco_data['images'][:5]:
    print(f"  {img['file_name']}")

# Count annotations per category
from collections import Counter
cat_counts = Counter(ann['category_id'] for ann in coco_data['annotations'])
print("\nAnnotations per category:")
for cat_id, count in sorted(cat_counts.items()):
    print(f"  Category {cat_id}: {count} annotations")