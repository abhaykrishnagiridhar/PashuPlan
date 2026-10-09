"""Command line: `python -m pashuplan.train [--out DIR]` trains both models and prints the held-out metrics."""
from __future__ import annotations

import argparse
import json

from pashuplan import models


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Train the PashuPlan models on simulated farms.")
    parser.add_argument("--out", default=str(models.DEFAULT_DIR), help="folder for the model JSON files")
    args = parser.parse_args(argv)
    metrics = models.train_and_save(args.out)
    print(json.dumps(metrics, indent=1))
    print(f"\nSaved models to {args.out}")
    return metrics


if __name__ == "__main__":
    main()
