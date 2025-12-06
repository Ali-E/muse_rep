#!/usr/bin/env python3
"""
Plot metrics across different inclusion values for a specific epoch.

This script reads CSV result files with naming pattern like:
    books_knowmem_f_some_npo_0.05_0.5.csv
    
The inclusion value is extracted from the filename after the last underscore.

Usage:
    python plot_metrics_by_inclusion.py <csv_file> <epoch_number> [--output <output_dir>]
    
Example:
    python plot_metrics_by_inclusion.py books_knowmem_f_some_npo_0.05_0.5.csv 3
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import re
from pathlib import Path
import numpy as np


def extract_inclusion_from_filename(filename):
    """Extract the inclusion value from filename pattern like 'prefix_0.5.csv'."""
    # Remove .csv extension
    name = Path(filename).stem
    # Get the last part after underscore
    parts = name.split('_')
    try:
        # Try to convert the last part to float (the inclusion value)
        inclusion = float(parts[-1])
        return inclusion
    except (ValueError, IndexError):
        return None


def plot_metrics_by_inclusion(csv_file, epoch_number, output_dir=None):
    """
    Read CSV file and plot metrics for a specific epoch across different inclusion values.
    
    Args:
        csv_file: Path to the CSV file containing results
        epoch_number: Epoch number to plot (int)
        output_dir: Directory to save plots (default: figures/)
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract inclusion value from filename
    file_inclusion = extract_inclusion_from_filename(csv_file)
    
    # Filter for the specific epoch
    if 'epoch' in df.columns:
        epoch_data = df[df['epoch'] == epoch_number].copy()
        if len(epoch_data) == 0:
            print(f"Error: No data found for epoch {epoch_number}")
            print(f"Available epochs: {sorted(df['epoch'].unique())}")
            return
    else:
        # If no epoch column, use all data
        epoch_data = df.copy()
        print(f"Warning: No 'epoch' column found. Using all data.")
    
    # Get metric columns (exclude metadata columns)
    exclude_cols = {'name', 'epoch', 'indices_seed', 'including_ratio'}
    metric_cols = [col for col in df.columns if col not in exclude_cols]
    
    if len(metric_cols) == 0:
        print("Error: No metric columns found in CSV file")
        return
    
    # Group by name (which contains the inclusion value pattern like npo_0.05)
    # Extract inclusion value from 'name' column
    def extract_inclusion_from_name(name):
        """Extract numeric value from name like 'npo_0.05'."""
        match = re.search(r'_(\d+\.?\d*)', name)
        if match:
            return float(match.group(1))
        return None
    
    epoch_data['inclusion_value'] = epoch_data['name'].apply(extract_inclusion_from_name)
    
    # If we couldn't extract from name, use including_ratio column if available
    if epoch_data['inclusion_value'].isna().all() and 'including_ratio' in epoch_data.columns:
        epoch_data['inclusion_value'] = epoch_data['including_ratio']
    
    # Remove rows where we couldn't extract inclusion value
    epoch_data = epoch_data.dropna(subset=['inclusion_value'])
    
    if len(epoch_data) == 0:
        print("Error: Could not extract inclusion values from data")
        return
    
    # Sort by inclusion value
    epoch_data = epoch_data.sort_values('inclusion_value')
    
    # Set up output directory
    if output_dir is None:
        output_dir = 'figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename for output
    base_filename = Path(csv_file).stem
    
    # Create plots for each metric
    n_metrics = len(metric_cols)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, metric in enumerate(metric_cols):
        ax = axes[idx]
        
        # Skip if all values are NaN
        if epoch_data[metric].isna().all():
            ax.set_visible(False)
            continue
        
        # Plot the metric
        x = epoch_data['inclusion_value']
        y = epoch_data[metric]
        
        # Handle extreme values (like perplexity explosion)
        if y.max() > 1e10:
            # Use log scale for y-axis
            ax.set_yscale('log')
        
        ax.plot(x, y, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel('Inclusion Value', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric} (Epoch {epoch_number})', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on points
        for xi, yi in zip(x, y):
            if not np.isnan(yi):
                if yi < 1e10:
                    ax.annotate(f'{yi:.3f}', (xi, yi), textcoords="offset points",
                               xytext=(0, 10), ha='center', fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(metric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, f'{base_filename}_epoch{epoch_number}_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {output_path}")
    
    # Also create individual plots for each metric (better for detailed viewing)
    for metric in metric_cols:
        if epoch_data[metric].isna().all():
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = epoch_data['inclusion_value']
        y = epoch_data[metric]
        
        # Handle extreme values
        if y.max() > 1e10:
            ax.set_yscale('log')
        
        ax.plot(x, y, marker='o', linewidth=2, markersize=10, color='steelblue')
        ax.set_xlabel('Inclusion Value', fontsize=14)
        ax.set_ylabel(metric, fontsize=14)
        ax.set_title(f'{metric} vs Inclusion Value (Epoch {epoch_number})', 
                    fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for xi, yi in zip(x, y):
            if not np.isnan(yi):
                if yi < 1e10:
                    ax.annotate(f'{yi:.3f}', (xi, yi), textcoords="offset points",
                               xytext=(0, 10), ha='center', fontsize=10)
        
        plt.tight_layout()
        
        # Save individual metric plot
        output_path = os.path.join(output_dir, 
                                   f'{base_filename}_epoch{epoch_number}_{metric}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")
        plt.close()
    
    print(f"\nSummary:")
    print(f"  CSV file: {csv_file}")
    print(f"  Epoch: {epoch_number}")
    print(f"  Number of inclusion values: {len(epoch_data)}")
    print(f"  Inclusion range: {epoch_data['inclusion_value'].min():.2f} - {epoch_data['inclusion_value'].max():.2f}")
    print(f"  Metrics plotted: {len(metric_cols)}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot metrics across different inclusion values for a specific epoch.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_metrics_by_inclusion.py books_knowmem_f_some_npo_0.05_0.5.csv 3
  python plot_metrics_by_inclusion.py results.csv 5 --output my_figures/
        """
    )
    
    parser.add_argument('csv_file', help='Path to CSV file containing results')
    parser.add_argument('epoch', type=int, help='Epoch number to plot')
    parser.add_argument('--output', '-o', default='figures', 
                       help='Output directory for plots (default: figures/)')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.csv_file):
        print(f"Error: File not found: {args.csv_file}")
        return 1
    
    plot_metrics_by_inclusion(args.csv_file, args.epoch, args.output)
    
    return 0


if __name__ == '__main__':
    exit(main())
