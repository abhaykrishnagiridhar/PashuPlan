"""Train, evaluate, save and load the two PashuPlan models.

Models are trained on farms with randomized timelines (seeds 100+) and judged on farms they
never saw (seeds 200+). The hero farm (seed 42) is in neither set.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pashuplan.attribution import LossAttribution
from pashuplan.data_gen import generate_farm
from pashuplan.risk_model import MastitisRiskModel

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "models"
FILES = {"attribution": "loss_attribution.json", "risk": "mastitis_risk.json"}
METRICS_FILE = "metrics.json"
TRAIN_SEEDS = range(100, 112)
TEST_SEEDS = range(201, 206)
HERO_SEED = 42
DATA_NOTE = ("All farms are simulated. Metrics show how well the models recover the simulator's causes "
             "on unseen simulated farms, not performance on real farms.")


@dataclass(frozen=True)
class Models:
    attribution: LossAttribution
    risk: MastitisRiskModel


def save(models: Models, directory: Path | str) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / FILES["attribution"]).write_text(json.dumps(models.attribution.to_dict(), indent=1), encoding="utf-8")
    (directory / FILES["risk"]).write_text(json.dumps(models.risk.to_dict(), indent=1), encoding="utf-8")


def _read(path: Path, kind: str, loader):
    try:
        return loader(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{kind} model file {path} is corrupt: {exc}") from exc


def load(directory: Path | str = DEFAULT_DIR) -> Models | None:
    """The saved models, or None if they have not been trained yet. A damaged file raises."""
    directory = Path(directory)
    paths = {k: directory / name for k, name in FILES.items()}
    if not all(p.exists() for p in paths.values()):
        return None
    return Models(attribution=_read(paths["attribution"], "loss_attribution", LossAttribution.from_dict),
                  risk=_read(paths["risk"], "mastitis_risk", MastitisRiskModel.from_dict))


def train_and_save(directory: Path | str = DEFAULT_DIR, train_seeds=TRAIN_SEEDS, test_seeds=TEST_SEEDS) -> dict:
    train_seeds, test_seeds = list(train_seeds), list(test_seeds)
    train = [generate_farm(seed=s, randomize_events=True) for s in train_seeds]
    test = [generate_farm(seed=s, randomize_events=True) for s in test_seeds]
    trained = Models(attribution=LossAttribution.fit(train), risk=MastitisRiskModel.fit(train))
    hero = generate_farm(seed=HERO_SEED)
    metrics = {
        "note": DATA_NOTE,
        "train_seeds": train_seeds,
        "test_seeds": test_seeds,
        "risk_held_out": trained.risk.evaluate(test),
        "risk_threshold": round(trained.risk.threshold, 3),
        "attribution_coefs": {k: round(v, 4) for k, v in trained.attribution.coefs.items()},
        "attribution_hero": trained.attribution.explain(hero),
        "early_warning_hero": [{"cow_id": w["cow_id"], "lead_days": w["lead_days"]}
                               for w in trained.risk.early_warnings(hero)],
    }
    save(trained, directory)
    (Path(directory) / METRICS_FILE).write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    return metrics
