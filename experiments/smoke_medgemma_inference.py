"""Smoke test de inferencia para la tarea de Pablo.

Ejemplos:
    python experiments/smoke_medgemma_inference.py --mode text --prompt "Explain glaucoma."
    python experiments/smoke_medgemma_inference.py --mode image-text --image sample.jpg
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.medgemma_runtime import MedGemmaRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MedGemma baseline or LoRA inference.")
    parser.add_argument("--model-id", default="google/medgemma-1.5-4b-it")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--mode", choices=["text", "image", "image-text"], default="image-text")
    parser.add_argument("--image", default=None, help="Path to an RGB image.")
    parser.add_argument(
        "--prompt",
        default="Describe the ophthalmological findings in this fundus image.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output", default="results/smoke_medgemma_inference.json")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()

    if args.mode in {"image", "image-text"} and not args.image:
        raise ValueError("--image is required for image and image-text modes.")

    prompt = "Describe this medical image." if args.mode == "image" else args.prompt
    image = args.image if args.mode in {"image", "image-text"} else None

    runtime = MedGemmaRuntime(
        {
            "model_id": args.model_id,
            "adapter_path": args.adapter_path,
            "max_new_tokens": args.max_new_tokens,
            "torch_dtype": "bfloat16",
            "device_map": "auto",
            "do_sample": False,
        }
    )
    text = runtime.generate(prompt=prompt, image=image)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "adapter_path": args.adapter_path,
        "mode": args.mode,
        "image": args.image,
        "prompt": prompt,
        "text": text,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(text)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
