"""Print the winning sweep configuration as shell-evalable assignments.

Reads ``results/sweep/summary.json`` (produced by ``aggregate_sweep``) and emits
the hyperparameters of the selected configuration so the final multi-seed test
job can train exactly the winner. Selector is one of ``overall``, ``generic`` or
``conditioned``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/sweep/summary.json")
    parser.add_argument(
        "--selector",
        choices=("overall", "generic", "conditioned"),
        default="overall",
    )
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    if args.selector == "overall":
        record = summary["best_overall"]
    else:
        record = summary["best_by_prompt_mode"][args.selector]

    if args.export:
        print(f"BEST_TAG={record['tag']}")
        print(f"PROMPT_MODE={record['prompt_mode']}")
        print(f"LORA_RANK={record['lora_rank']}")
        print(f"LORA_ALPHA={record['lora_alpha']}")
        print(f"LEARNING_RATE={record['learning_rate']}")
        print(f"NUM_EPOCHS={record['num_epochs']}")
    else:
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
