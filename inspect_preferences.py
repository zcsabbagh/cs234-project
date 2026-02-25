"""Quick script to inspect None winner/loser counts in preferences.jsonl."""

import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "preferences.jsonl"

none_count = 0
total = 0
with open(path) as f:
    for line in f:
        total += 1
        row = json.loads(line)
        if row.get("winner") is None or row.get("loser") is None:
            none_count += 1

print(f"Total rows: {total}")
print(f"Rows with None winner/loser: {none_count}")
print(f"Valid rows (for training): {total - none_count}")
