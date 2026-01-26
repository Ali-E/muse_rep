#!/bin/bash

# Aggregate cross-dataset PS scores by author pairs
# This script processes the output from run_compute_ps_cross_dataset_parallel.sh
# and creates author-level rankings for each site type.

# Configuration
OUTPUT_DIR="ps_cross_dataset_outputs_short"
TARGET_AUTHORS_CSV="tofu_data/authors_paragraphs_short.csv"
SOURCE_AUTHORS_CSV="tofu_data/source_authors.csv"
SAMPLES_PER_AUTHOR=20

# Site types to process (should match what was used in run_compute_ps_cross_dataset_parallel.sh)
SITE_TYPES=("mlp" "resid" "overall")

echo "======================================"
echo "Aggregating PS scores by author pairs"
echo "======================================"
echo "Output directory: $OUTPUT_DIR"
echo "Target authors: $TARGET_AUTHORS_CSV"
echo "Source authors: $SOURCE_AUTHORS_CSV"
echo "Samples per author: $SAMPLES_PER_AUTHOR"
echo "Site types: ${SITE_TYPES[@]}"
echo "======================================"

for SITE_TYPE in "${SITE_TYPES[@]}"; do
    echo ""
    echo "Processing site type: $SITE_TYPE"
    echo "--------------------------------------"

    DETAILED_CSV="${OUTPUT_DIR}/ps_detailed_${SITE_TYPE}.csv"
    AVG_RANKED_CSV="${OUTPUT_DIR}/ps_avg_ranked_${SITE_TYPE}.csv"

    # Check if detailed file exists
    if [ ! -f "$DETAILED_CSV" ]; then
        echo "Warning: Detailed file not found: $DETAILED_CSV, skipping"
        continue
    fi

    # Check file is not empty (just header)
    NUM_LINES=$(wc -l < "$DETAILED_CSV")
    if [ "$NUM_LINES" -le 1 ]; then
        echo "Warning: Detailed file is empty: $DETAILED_CSV, skipping"
        continue
    fi

    # Run aggregation using detailed CSV
    echo "Aggregating from: $DETAILED_CSV"
    python3 aggregate_by_author.py \
        --detailed_csv="$DETAILED_CSV" \
        --target_authors_csv="$TARGET_AUTHORS_CSV" \
        --source_authors_csv="$SOURCE_AUTHORS_CSV" \
        --samples_per_author="$SAMPLES_PER_AUTHOR" \
        --out_author_scores_csv="${OUTPUT_DIR}/author_scores_${SITE_TYPE}.csv" \
        --out_author_ranked_csv="${OUTPUT_DIR}/author_ranked_${SITE_TYPE}.csv"

    # Also run with avg_ranked if it exists and has data
    if [ -f "$AVG_RANKED_CSV" ]; then
        NUM_LINES=$(wc -l < "$AVG_RANKED_CSV")
        if [ "$NUM_LINES" -gt 1 ]; then
            echo "Also aggregating from avg_ranked: $AVG_RANKED_CSV"
            python3 aggregate_by_author.py \
                --detailed_csv="$DETAILED_CSV" \
                --avg_ranked_csv="$AVG_RANKED_CSV" \
                --target_authors_csv="$TARGET_AUTHORS_CSV" \
                --source_authors_csv="$SOURCE_AUTHORS_CSV" \
                --samples_per_author="$SAMPLES_PER_AUTHOR" \
                --out_author_scores_csv="${OUTPUT_DIR}/author_avg_scores_${SITE_TYPE}.csv" \
                --out_author_ranked_csv="${OUTPUT_DIR}/author_avg_ranked_${SITE_TYPE}.csv"
        fi
    fi

    echo "Site type $SITE_TYPE complete!"
done

echo ""
echo "======================================"
echo "All aggregations complete!"
echo "======================================"
echo "Output files:"
for SITE_TYPE in "${SITE_TYPES[@]}"; do
    if [ -f "${OUTPUT_DIR}/author_ranked_${SITE_TYPE}.csv" ]; then
        echo "  ${OUTPUT_DIR}/author_scores_${SITE_TYPE}.csv"
        echo "  ${OUTPUT_DIR}/author_ranked_${SITE_TYPE}.csv"
    fi
    if [ -f "${OUTPUT_DIR}/author_avg_ranked_${SITE_TYPE}.csv" ]; then
        echo "  ${OUTPUT_DIR}/author_avg_scores_${SITE_TYPE}.csv"
        echo "  ${OUTPUT_DIR}/author_avg_ranked_${SITE_TYPE}.csv"
    fi
done
echo "======================================"
