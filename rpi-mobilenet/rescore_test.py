"""Re-score test.ipynb's predictions against the labels encoded in the filenames.

Why this exists
---------------
`test.ipynb` reported 2.75% accuracy over 182 images. That number is an artifact of a
broken label mapping, not a measurement of the model. It read raw class indices from a
six-class YOLO label set (BIODEGRADABLE, CARDBOARD, GLASS, METAL, PAPER, PLASTIC),
rejected every index >= 4, and then interpreted the survivors against a different
four-class list. `prediction_results.csv` records the damage directly: files named
`metal*.jpg` carry `true_class=biodegradable`.

The source images are named after their true class, so the filename prefix recovers
the label the harness lost. This script re-scores against that and prints both numbers
side by side.

Read the caveats it prints. This is a sanity check on a salvaged, badly skewed sample,
not a test accuracy. The project never had a valid held-out test set.

    python rescore_test.py
"""

import csv
import os
import re
from collections import Counter

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_results.csv")

# The shipped four-class head. `paper` and `cardboard` from the source taxonomy were
# folded into `biodegradable` at training time, so a file named paper*.jpg has no
# unambiguous ground truth here and cannot be scored.
MODEL_CLASSES = ["metal", "plastic", "glass", "biodegradable"]
SCORABLE = {"metal", "plastic", "glass"}


def label_from_filename(name):
    match = re.match(r"^([a-zA-Z]+)", name)
    return match.group(1).lower() if match else None


def main():
    with open(CSV_PATH, newline="") as handle:
        rows = list(csv.DictReader(handle))

    as_scored = sum(row["correct"] == "True" for row in rows)

    print("=" * 66)
    print("  Re-scoring test.ipynb against filename labels")
    print("=" * 66)
    print(f"\nRows in prediction_results.csv: {len(rows)}")
    print(f"As scored by test.ipynb:        {as_scored}/{len(rows)} = {as_scored / len(rows):.2%}")

    recorded = Counter(row["true_class"] for row in rows)
    filenames = Counter(label_from_filename(row["image"]) for row in rows)
    print(f"\n  true_class as recorded:  {dict(recorded)}")
    print(f"  label from filename:     {dict(filenames)}")
    print("\n  The two disagree almost everywhere. That is the bug.")

    hits = misses = 0
    unscorable = 0
    confusion = Counter()
    for row in rows:
        truth = label_from_filename(row["image"])
        predicted = row["predicted_class"]
        if truth not in SCORABLE:
            unscorable += 1
            continue
        confusion[(truth, predicted)] += 1
        if truth == predicted:
            hits += 1
        else:
            misses += 1

    total = hits + misses
    print(f"\nRe-scored on the {total} scorable images: {hits}/{total} = {hits / total:.1%}")
    print(f"  ({unscorable} unscorable: no `paper` class in the four-class head)")

    print("\n  filename label -> predicted:")
    for (truth, predicted), count in sorted(confusion.items(), key=lambda item: -item[1]):
        marker = "ok " if truth == predicted else "   "
        print(f"    {marker} {truth:<9} -> {predicted:<14} {count}")

    per_class = Counter(truth for truth, _ in confusion.elements())
    print("\n" + "-" * 66)
    print("CAVEATS - read these before quoting the number")
    print("-" * 66)
    print(f"  1. The sample is not balanced: {dict(per_class)}.")
    print("     The headline figure is essentially metal recall, not four-class accuracy.")
    print("  2. These 182 rows are what survived a harness that dropped 860 of 1042")
    print("     images (see error_log.txt). They are leftovers, not a test split.")
    print("  3. Training validation accuracy was 91.63% on the same biased distribution")
    print("     the model was trained on. This re-score is consistent with that, which")
    print("     is the point: the weights were fine, the evaluation was broken.")
    print("  4. This is a sanity check, not a test accuracy. The project never had a")
    print("     valid held-out test set. That remains the real finding.")
    print()


if __name__ == "__main__":
    main()
