"""
Download TOFU dataset from HuggingFace Hub and save as CSV files.

Usage:
    python download_tofu_dataset.py --subset full --output_dir tofu_data
    python download_tofu_dataset.py --subset forget10 --output_dir tofu_data
"""

import argparse
import os
from datasets import load_dataset
import pandas as pd


def download_tofu_dataset(subset: str = "full", output_dir: str = "tofu_data"):
    """
    Download TOFU dataset and save as CSV files.
    
    Args:
        subset: TOFU subset (e.g., "full", "forget01", "forget05", "forget10")
        output_dir: Directory to save CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading TOFU dataset (subset: {subset})...")
    
    # Download all splits
    splits = ["train", "validation", "test"]
    
    for split in splits:
        try:
            print(f"\nDownloading {split} split...")
            ds = load_dataset("locuslab/TOFU", subset, split=split)
            
            # Convert to pandas DataFrame
            df = ds.to_pandas()
            
            # Save as CSV
            output_file = os.path.join(output_dir, f"tofu_{subset}_{split}.csv")
            df.to_csv(output_file, index=False)
            
            print(f"Saved {len(df)} examples to {output_file}")
            print(f"Columns: {list(df.columns)}")
            
        except Exception as e:
            print(f"Error downloading {split} split: {e}")
    
    print(f"\nAll files saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Download TOFU dataset from HuggingFace Hub and save as CSV"
    )
    
    parser.add_argument(
        "--subset",
        type=str,
        default="full",
        help="TOFU subset to download (default: 'full', options: 'forget01', 'forget05', 'forget10', etc.)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tofu_data",
        help="Directory to save CSV files (default: 'tofu_data')"
    )
    parser.add_argument(
        "--all_subsets",
        action="store_true",
        help="Download all TOFU subsets"
    )
    
    args = parser.parse_args()
    
    if args.all_subsets:
        # Download all common TOFU subsets
        subsets = ["full", "forget01", "forget05", "forget10"]
        for subset in subsets:
            print(f"\n{'='*60}")
            print(f"Processing subset: {subset}")
            print(f"{'='*60}")
            download_tofu_dataset(subset=subset, output_dir=args.output_dir)
    else:
        download_tofu_dataset(subset=args.subset, output_dir=args.output_dir)
    
    print("\n" + "="*60)
    print("Download complete!")
    print("="*60)


if __name__ == "__main__":
    main()
