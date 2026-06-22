"""Single source of truth for the experimental MedGemma-LoRA sweep grid.

The grid is a full factorial over LoRA rank, learning rate, training epochs and
prompt mode. ``lora_alpha`` follows the common ``alpha = 2 * rank`` convention
(LoRA, Hu et al. 2021; QLoRA, Dettmers et al. 2023), so the effective scaling is
held constant across ranks and only capacity changes.

A SLURM array job maps ``SLURM_ARRAY_TASK_ID`` to one configuration via
``--index``; ``--export`` prints shell-evalable variable assignments.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

# Reduced grid: rank and learning rate fixed at literature-sane defaults; only
# the training duration (epochs) and prompt mode are searched on validation.
RANKS = (8,)
LEARNING_RATES = (1e-4,)
EPOCHS = (3, 6, 9)
PROMPT_MODES = ("generic", "conditioned")


@dataclass(frozen=True)
class SweepPoint:
    index: int
    prompt_mode: str
    lora_rank: int
    lora_alpha: int
    learning_rate: float
    num_epochs: int

    @property
    def tag(self) -> str:
        lr = f"{self.learning_rate:.0e}".replace("-0", "-")
        return (
            f"{self.prompt_mode}_r{self.lora_rank}"
            f"_lr{lr}_ep{self.num_epochs}"
        )


def build_grid() -> list[SweepPoint]:
    points: list[SweepPoint] = []
    index = 0
    for prompt_mode in PROMPT_MODES:
        for rank in RANKS:
            for lr in LEARNING_RATES:
                for epochs in EPOCHS:
                    points.append(
                        SweepPoint(
                            index=index,
                            prompt_mode=prompt_mode,
                            lora_rank=rank,
                            lora_alpha=2 * rank,
                            learning_rate=lr,
                            num_epochs=epochs,
                        )
                    )
                    index += 1
    return points


def get_point(index: int) -> SweepPoint:
    grid = build_grid()
    if not 0 <= index < len(grid):
        raise IndexError(
            f"Sweep index {index} out of range [0, {len(grid) - 1}]"
        )
    return grid[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--index", type=int)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    if args.count:
        print(len(build_grid()))
        return
    if args.index is None:
        for point in build_grid():
            print(f"{point.index:2d}  {point.tag}")
        return

    point = get_point(args.index)
    if args.export:
        print(f"SWEEP_TAG={point.tag}")
        print(f"PROMPT_MODE={point.prompt_mode}")
        print(f"LORA_RANK={point.lora_rank}")
        print(f"LORA_ALPHA={point.lora_alpha}")
        print(f"LEARNING_RATE={point.learning_rate}")
        print(f"NUM_EPOCHS={point.num_epochs}")
    else:
        print(point)


if __name__ == "__main__":
    main()
