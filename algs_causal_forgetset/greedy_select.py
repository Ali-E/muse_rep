import argparse
import csv
from typing import Dict, List, Set


def load_weights(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser(description="Greedy core-set selection using per-prompt weights w_p(s)")
    ap.add_argument("--weights_csv", required=True, help="Detailed weights: sample_id,question_id,weight")
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    rows = load_weights(args.weights_csv)

    # Build prompt universe and per-sample weights
    prompts: Set[str] = set()
    by_sample: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        sid = r["sample_id"]
        pid = r["question_id"]
        w = float(r.get("weight", "0"))
        prompts.add(pid)
        by_sample.setdefault(sid, []).append({"pid": pid, "w": w})

    # Coverage state
    R: Dict[str, float] = {pid: 0.0 for pid in prompts}
    selected: List[str] = []
    remaining: Set[str] = set(by_sample.keys())

    for _ in range(args.budget):
        best_sid = None
        best_gain = -1.0
        for sid in list(remaining):
            gain = 0.0
            for e in by_sample[sid]:
                pid = e["pid"]
                w = e["w"]
                gain += min(w, 1.0 - R[pid])
            if gain > best_gain:
                best_gain = gain
                best_sid = sid
        if best_sid is None or best_gain <= 0.0:
            break
        # Select and update residuals
        selected.append(best_sid)
        remaining.remove(best_sid)
        for e in by_sample[best_sid]:
            pid = e["pid"]
            w = e["w"]
            R[pid] = min(1.0, R[pid] + w)

    # Write output
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "sample_id", "marginal_gain"])
        writer.writeheader()
        # Recompute marginal gains for reporting
        # (optional; here we report 0 for simplicity)
        for i, sid in enumerate(selected, start=1):
            writer.writerow({"rank": i, "sample_id": sid, "marginal_gain": ""})
    print(f"Wrote {len(selected)} selected samples to {args.out_csv}")


if __name__ == "__main__":
    main()


