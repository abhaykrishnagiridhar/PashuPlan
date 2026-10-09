import json

import pytest

from pashuplan import models as model_store


def test_save_then_load_gives_identical_predictions(models, farm, tmp_path):
    model_store.save(models, tmp_path)
    loaded = model_store.load(tmp_path)
    assert loaded.attribution.explain(farm) == models.attribution.explain(farm)
    assert loaded.risk.watch_list(farm) == models.risk.watch_list(farm)


def test_load_returns_none_when_nothing_was_trained(tmp_path):
    assert model_store.load(tmp_path / "missing") is None


def test_load_refuses_a_corrupt_file_instead_of_guessing(models, tmp_path):
    model_store.save(models, tmp_path)
    (tmp_path / model_store.FILES["risk"]).write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="mastitis_risk"):
        model_store.load(tmp_path)


def test_train_and_save_writes_models_and_honest_metrics(tmp_path):
    metrics = model_store.train_and_save(tmp_path, train_seeds=range(100, 103), test_seeds=(201,))
    for name in (*model_store.FILES.values(), "metrics.json"):
        assert (tmp_path / name).exists()
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved == json.loads(json.dumps(metrics))
    assert set(saved["risk_held_out"]) >= {"auc", "precision", "recall", "alerts_per_cow_week"}
    assert saved["test_seeds"] == [201] and saved["train_seeds"] == [100, 101, 102]
    assert not set(saved["train_seeds"]) & set(saved["test_seeds"])
    assert set(saved["attribution_hero"]["causes"]) == {"heat", "feed", "sick"}


def test_train_command_line_passes_the_output_folder_and_prints_metrics(monkeypatch, capsys, tmp_path):
    from pashuplan import train
    seen = {}
    monkeypatch.setattr(model_store, "train_and_save", lambda out: seen.setdefault("out", out) and {"auc": 0.9})
    train.main(["--out", str(tmp_path)])
    assert seen["out"] == str(tmp_path)
    assert '"auc": 0.9' in capsys.readouterr().out


def test_shipped_models_load_and_work_on_the_hero_farm(farm):
    shipped = model_store.load(model_store.DEFAULT_DIR)
    assert shipped is not None, "run `python -m pashuplan.train` to create the models folder"
    assert shipped.risk.watch_list(farm)[0]["cow_id"] == farm.ground_truth["at_risk_cows"][0]
    assert all(v > 5 for v in shipped.attribution.explain(farm)["causes"].values())
