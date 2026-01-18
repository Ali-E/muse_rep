"""
Aggregate cross-dataset PS scores by author pairs.

For each (source_author, target_author) pair, compute an average score
over all (source_sample, target_sample) pairs, then rank source authors
for each target author.

Source author is derived from source_sample_id: author_idx = sample_id // 20
Target author is derived from target_prompt_id using authors_paragraphs_final.csv
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_source_author(sample_id: int, samples_per_author: int = 20) -> int:
    """Get source author index from sample_id."""
    return sample_id // samples_per_author


def load_target_author_mapping(authors_csv: str) -> Dict[str, str]:
    """Load mapping from target_prompt_id to author name."""
    rows = load_csv(authors_csv)
    return {str(r["id"]): r["author"] for r in rows}


def load_source_author_names(tofu_train_csv: str, samples_per_author: int = 20) -> Dict[int, str]:
    """Load mapping from source author index to author name.

    Author name is extracted from the first row of each author's samples.
    """
    rows = load_csv(tofu_train_csv)
    author_names: Dict[int, str] = {}

    for i, row in enumerate(rows):
        author_idx = i // samples_per_author
        if author_idx not in author_names:
            # Extract author name from the answer text
            # The answer typically contains the author name
            answer = row.get("answer", "")
            # Try to extract name from common patterns
            # e.g., "The author in question is Jaime Vasquez, an esteemed..."
            # or "Chukwu Akabueze is the biographical writer..."
            question = row.get("question", "")

            # Use the first meaningful name found
            # For simplicity, we'll use author index as fallback
            author_names[author_idx] = f"Author_{author_idx}"

            # Try to extract from question if it mentions a specific name
            # This is a heuristic - adjust based on actual data format

    return author_names


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate cross-dataset PS scores by author pairs."
    )
    ap.add_argument(
        "--detailed_csv",
        required=True,
        help="Input detailed CSV from compute_ps_cross_dataset.py (ps_detailed_*.csv)",
    )
    ap.add_argument(
        "--avg_ranked_csv",
        default=None,
        help="Input average-ranked CSV from compute_ps_cross_dataset.py (ps_avg_ranked_*.csv)",
    )
    ap.add_argument(
        "--target_authors_csv",
        required=True,
        help="CSV with id,author columns mapping target_prompt_id to author name",
    )
    ap.add_argument(
        "--source_authors_csv",
        default=None,
        help="Optional CSV with author names for source. If not provided, uses Author_N naming.",
    )
    ap.add_argument(
        "--samples_per_author",
        type=int,
        default=20,
        help="Number of samples per author in source data (default: 20)",
    )
    ap.add_argument(
        "--out_author_scores_csv",
        required=True,
        help="Output CSV with average scores per (source_author, target_author) pair",
    )
    ap.add_argument(
        "--out_author_ranked_csv",
        required=True,
        help="Output CSV with source authors ranked per target author",
    )
    ap.add_argument(
        "--score_column",
        default="fraction_restored",
        help="Column to use for scoring (default: fraction_restored). Use 'avg_fraction_restored' for avg_ranked input.",
    )
    args = ap.parse_args()

    # Load target author mapping
    target_author_map = load_target_author_mapping(args.target_authors_csv)
    print(f"Loaded {len(target_author_map)} target author mappings")

    # Load source author names if provided
    source_author_names: Dict[int, str] = {}
    if args.source_authors_csv:
        # Load from CSV if provided
        rows = load_csv(args.source_authors_csv)
        for r in rows:
            idx = int(r.get("author_idx", r.get("id", -1)))
            name = r.get("author", r.get("name", f"Author_{idx}"))
            source_author_names[idx] = name

    # Load detailed scores
    if args.avg_ranked_csv and os.path.exists(args.avg_ranked_csv):
        print(f"Using avg_ranked CSV: {args.avg_ranked_csv}")
        detailed_rows = load_csv(args.avg_ranked_csv)
        score_col = "avg_fraction_restored"
    else:
        print(f"Using detailed CSV: {args.detailed_csv}")
        detailed_rows = load_csv(args.detailed_csv)
        score_col = args.score_column

    print(f"Loaded {len(detailed_rows)} detailed rows")
    print(f"Using score column: {score_col}")

    # Aggregate scores by (source_author, target_author) pair
    # Structure: {(source_author_idx, target_author): [scores]}
    author_pair_scores: Dict[Tuple[int, str], List[float]] = defaultdict(list)

    # Also track sample counts for debugging
    sample_counts: Dict[Tuple[int, str], int] = defaultdict(int)

    for row in detailed_rows:
        source_sample_id = int(row["source_sample_id"])
        target_prompt_id = str(row["target_prompt_id"])

        # Get source author
        source_author_idx = get_source_author(source_sample_id, args.samples_per_author)

        # Get target author
        target_author = target_author_map.get(target_prompt_id)
        if target_author is None:
            continue

        # Get score
        try:
            score = float(row[score_col])
        except (KeyError, ValueError):
            continue

        key = (source_author_idx, target_author)
        author_pair_scores[key].append(score)
        sample_counts[key] += 1

        # Track source author name if not already known
        if source_author_idx not in source_author_names:
            source_question = row.get("source_question", "")
            source_author_names[source_author_idx] = f"Author_{source_author_idx}"

    print(f"Found {len(author_pair_scores)} unique (source_author, target_author) pairs")

    # Compute average scores per pair
    author_scores_rows = []
    for (source_idx, target_author), scores in author_pair_scores.items():
        avg_score = sum(scores) / len(scores)
        source_name = source_author_names.get(source_idx, f"Author_{source_idx}")

        author_scores_rows.append({
            "source_author_idx": source_idx,
            "source_author": source_name,
            "target_author": target_author,
            "avg_score": avg_score,
            "n_samples": len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
        })

    # Write author scores
    with open(args.out_author_scores_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_author_idx", "source_author", "target_author",
                "avg_score", "n_samples", "min_score", "max_score"
            ],
        )
        writer.writeheader()
        writer.writerows(author_scores_rows)

    print(f"Wrote {len(author_scores_rows)} author pair scores to {args.out_author_scores_csv}")

    # Create rankings: for each target author, rank source authors by avg_score
    target_rankings: Dict[str, List[Dict]] = defaultdict(list)
    for row in author_scores_rows:
        target_rankings[row["target_author"]].append(row)

    ranked_rows = []
    for target_author in sorted(target_rankings.keys()):
        rankings = target_rankings[target_author]
        # Sort by avg_score descending (higher = better restoration)
        rankings.sort(key=lambda x: x["avg_score"], reverse=True)

        for rank, entry in enumerate(rankings, start=1):
            ranked_rows.append({
                "target_author": target_author,
                "rank": rank,
                "source_author_idx": entry["source_author_idx"],
                "source_author": entry["source_author"],
                "avg_score": entry["avg_score"],
                "n_samples": entry["n_samples"],
            })

    # Write ranked output
    with open(args.out_author_ranked_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target_author", "rank", "source_author_idx", "source_author",
                "avg_score", "n_samples"
            ],
        )
        writer.writeheader()
        writer.writerows(ranked_rows)

    print(f"Wrote {len(ranked_rows)} ranked rows to {args.out_author_ranked_csv}")

    # Print summary
    print("\n" + "=" * 60)
    print("Top 3 source authors per target author:")
    print("=" * 60)
    for target_author in sorted(target_rankings.keys())[:10]:
        print(f"\nTarget: {target_author}")
        for row in ranked_rows:
            if row["target_author"] == target_author and row["rank"] <= 3:
                print(f"  Rank {row['rank']}: {row['source_author']} "
                      f"(idx={row['source_author_idx']}, avg={row['avg_score']:.4f}, n={row['n_samples']})")


if __name__ == "__main__":
    main()
