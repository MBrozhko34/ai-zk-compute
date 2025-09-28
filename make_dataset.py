#!/usr/bin/env python3
import csv, os, argparse, random
from typing import List, Tuple

# Label rule the model can learn (same as before)
# y = 1 if x0 + 0.6*x1 > 0.9  else 0
def label(x0: float, x1: float) -> int:
    return 1 if (x0 + 0.6 * x1) > 0.9 else 0

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def rnd2(rng: random.Random, x: float) -> float:
    # two decimals for readability; quantization later is still 0..15 half-up
    return round(clamp01(x), 2)

def sample_pair(rng: random.Random) -> Tuple[float, float]:
    """
    Mixture sampler:
      - 70% uniform over [0,1]^2
      - 30% concentrated near the decision boundary x0 + 0.6*x1 = 0.9
        (this drives variability across hyperparameters without changing your code)
    """
    if rng.random() < 0.30:
        # choose x1, place x0 near boundary with small noise
        x1 = rng.random()
        base_x0 = 0.9 - 0.6 * x1
        # noise band around the boundary; tuned to keep many points near decision surface
        x0 = base_x0 + rng.uniform(-0.10, 0.10)
        return rnd2(rng, x0), rnd2(rng, x1)
    else:
        return rnd2(rng, rng.random()), rnd2(rng, rng.random())

def gen_rows(n: int, rng: random.Random, flip_prob: float) -> List[Tuple[float, float, int]]:
    rows: List[Tuple[float, float, int]] = []
    for _ in range(n):
        x0, x1 = sample_pair(rng)
        y = label(x0, x1)
        if flip_prob > 0 and rng.random() < flip_prob:
            y ^= 1
        rows.append((x0, x1, y))
    return rows

def write_csv(path: str, rows: List[Tuple[float, float, int]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x0", "x1", "y"])
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser(description="Generate learnable train/holdout CSVs for ai-zk-compute.")
    ap.add_argument("--train", type=int, default=1024, help="number of training rows (default: 1024)")
    ap.add_argument("--holdout", type=int, default=256, help="number of holdout rows (default: 256)")
    ap.add_argument("--seed", type=int, default=1337, help="PRNG seed (default: 1337)")
    ap.add_argument("--flip-train", type=float, default=0.03, help="label flip probability on train (default: 0.03)")
    ap.add_argument("--flip-holdout", type=float, default=0.00, help="label flip probability on holdout (default: 0.00)")
    ap.add_argument("--train-path", default="client/train.csv", help="path to write train CSV")
    ap.add_argument("--holdout-path", default="client/dataset.csv", help="path to write holdout CSV")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    train_rows   = gen_rows(args.train,   rng, args.flip_train)
    holdout_rows = gen_rows(args.holdout, rng, args.flip_holdout)

    write_csv(args.train_path,   train_rows)
    write_csv(args.holdout_path, holdout_rows)

    # quick summary
    tr_pos = sum(y for _, _, y in train_rows)
    te_pos = sum(y for _, _, y in holdout_rows)
    print(f"Wrote {args.train_path}  ({len(train_rows)} rows, positives={tr_pos})")
    print(f"Wrote {args.holdout_path} ({len(holdout_rows)} rows, positives={te_pos})")
    print("Reminder: run `make req` afterwards to open a NEW request id for these CSVs.")

if __name__ == "__main__":
    main()
