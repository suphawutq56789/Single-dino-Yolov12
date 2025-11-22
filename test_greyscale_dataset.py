#!/usr/bin/env python3
"""Test script to verify greyscale dataset loader."""

import sys
sys.path.insert(0, '.')

from ultralytics.data.triple_dataset_greyscale import TripleInputDatasetGreyscale
from ultralytics.cfg import get_cfg
from pathlib import Path

# Test dataset loading
print("Testing TripleInputDatasetGreyscale...")
print("=" * 60)

img_path = r"C:\Users\USER\Desktop\Modify code to grey scale\DATASET NEW\images\primary\train"

try:
    # Create simple hyp object
    hyp = get_cfg()

    # Create dataset
    dataset = TripleInputDatasetGreyscale(
        img_path=img_path,
        imgsz=224,
        cache=False,
        augment=False,
        hyp=hyp,
        prefix="test: ",
        rect=False,
        batch_size=2,
        stride=32,
        pad=0.0,
        single_cls=False,
        classes=None,
        fraction=1.0,
        data=None,
        task="detect"
    )

    print(f"✓ Dataset created successfully")
    print(f"  Total images: {len(dataset)}")
    print(f"  Triple input mode: {dataset.is_triple_input}")

    # Load first image
    if len(dataset) > 0:
        print("\nLoading first image...")
        img, (h0, w0), (h, w) = dataset.load_image(0, rect_mode=False)

        print(f"✓ Image loaded successfully")
        print(f"  Original size: {(h0, w0)}")
        print(f"  Current size: {(h, w)}")
        print(f"  Image shape: {img.shape}")
        print(f"  Image dtype: {img.dtype}")
        print(f"  Image min/max: {img.min()}/{img.max()}")

        expected_channels = 3
        actual_channels = img.shape[2] if len(img.shape) == 3 else 1

        if actual_channels == expected_channels:
            print(f"\n✅ SUCCESS! Image has {actual_channels} channels (greyscale triple input)")
        else:
            print(f"\n❌ FAILED! Expected {expected_channels} channels but got {actual_channels}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
