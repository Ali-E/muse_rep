#!/bin/bash

# Test the filtering logic in the shell script
SOURCE_SAMPLES_CSV="site_outputs_tofu_llama2/samples.csv"
OUTPUT_DIR="ps_cross_dataset_outputs"
NUM_GPUS=2
LIMIT=0
SITE_TYPES="mlp,resid"

python3 << EOF
import pandas as pd
import os

source_samples_csv = "$SOURCE_SAMPLES_CSV"
output_dir = "$OUTPUT_DIR"
num_gpus = $NUM_GPUS
limit = $LIMIT
site_types_arg = "$SITE_TYPES"

# Read the source samples CSV
if not os.path.exists(source_samples_csv):
    print(f"Error: Source samples CSV not found: {source_samples_csv}")
    exit(1)

df = pd.read_csv(source_samples_csv)

# Filter by site type if specified
if site_types_arg and site_types_arg.lower() != "all":
    # Parse comma-separated site types
    requested_types = [t.strip() for t in site_types_arg.split(',')]
    
    # Validate site types
    valid_types = ['overall', 'mlp', 'resid', 'last_mlp_in', 'last_mlp_out']
    invalid_types = [t for t in requested_types if t not in valid_types]
    if invalid_types:
        print(f"Error: Invalid site types: {invalid_types}")
        print(f"Valid types: {valid_types}")
        exit(1)
    
    # Filter dataframe
    df = df[df['variant'].isin(requested_types)]
    print(f"Filtered to site types: {requested_types}")
    print(f"Remaining rows after filtering: {len(df)}")

# Get unique sample_ids
sample_ids = df['sample_id'].unique().tolist()
print(f"Found {len(sample_ids)} unique source sample IDs")
print(f"Variants in filtered data: {sorted(df['variant'].unique())}")
EOF
