"""Tests for the built-in weight/calibration registry and the on-disk loaders.

`load_weights` and `load_calibration` are the seam between the shipped package data and
every scorer, so the registry entries are exercised against the real files rather than a
stubbed loader: a name in the map that has no matching file is a shipping bug that a mock
would hide.
"""

import json

import pytest
from conftest import epr_calibration, wepr_weights, write_json
from hypothesis import HealthCheck, given, settings

from artefactual.utils.io import (
    MODEL_CALIBRATION_MAP,
    MODEL_WEIGHT_MAP,
    load_calibration,
    load_weights,
)

# --- the shipped registry --------------------------------------------------------------


@pytest.mark.parametrize("model_name", sorted(MODEL_WEIGHT_MAP))
def test_every_registered_weight_name_loads(model_name):
    # a name users can pass must resolve to a file that ships in the package
    weights = load_weights(model_name)
    assert "intercept" in weights
    assert weights["coefficients"]


@pytest.mark.parametrize("model_name", sorted(MODEL_CALIBRATION_MAP))
def test_every_registered_calibration_name_loads(model_name):
    calibration = load_calibration(model_name)
    assert "intercept" in calibration
    assert calibration["coefficients"]


@pytest.mark.parametrize("model_name", sorted(MODEL_WEIGHT_MAP))
def test_registered_weights_are_wepr_shaped(model_name):
    # WEPR consumers index mean_rank_i/max_rank_i from 1..k; a gap silently zero-fills
    coefficients = load_weights(model_name)["coefficients"]
    k = sum(1 for key in coefficients if key.startswith("mean_rank_"))
    assert k > 0
    for rank in range(1, k + 1):
        assert f"mean_rank_{rank}" in coefficients
        assert f"max_rank_{rank}" in coefficients


@pytest.mark.parametrize("model_name", sorted(MODEL_CALIBRATION_MAP))
def test_registered_calibrations_are_epr_shaped(model_name):
    assert "mean_entropy" in load_calibration(model_name)["coefficients"]


# --- loading from a path ---------------------------------------------------------------


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(payload=wepr_weights())
def test_weights_round_trip_through_a_file(tmp_path, payload):
    # one file, rewritten per example — a fresh tmp dir per example would leave hundreds behind
    path = write_json(tmp_path, "w.json", payload)
    assert load_weights(path) == payload


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(payload=epr_calibration())
def test_calibration_round_trips_through_a_file(tmp_path, payload):
    path = write_json(tmp_path, "c.json", payload)
    assert load_calibration(str(path)) == payload


def test_load_weights_accepts_str_and_path_alike(tmp_path):
    path = write_json(tmp_path, "w.json", {"intercept": 1.0, "coefficients": {"mean_entropy": 2.0}})
    assert load_weights(path) == load_weights(str(path))


def test_load_calibration_accepts_a_path_object(tmp_path):
    # documented as `str`, but callers hold Paths; Path(identifier) makes this work
    path = write_json(tmp_path, "c.json", {"intercept": 1.0, "coefficients": {"mean_entropy": 2.0}})
    assert load_calibration(path) == load_calibration(str(path))


def test_registry_name_wins_over_an_identically_named_local_file(tmp_path, monkeypatch):
    # a stray file named after a model must not shadow the shipped calibration
    monkeypatch.chdir(tmp_path)
    name = "mistralai/Ministral-8B-Instruct-2410"
    shadow = {"intercept": 999.0, "coefficients": {"mean_entropy": 999.0}}
    local = tmp_path / name
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(shadow), encoding="utf-8")

    assert load_calibration(name) != shadow


# --- failure modes ---------------------------------------------------------------------


@pytest.mark.parametrize("loader", [load_weights, load_calibration])
def test_unknown_identifier_names_the_supported_models(loader):
    with pytest.raises(ValueError, match="mistralai/Ministral-8B-Instruct-2410"):
        loader("definitely-not-a-model")


@pytest.mark.parametrize("loader", [load_weights, load_calibration])
def test_malformed_json_is_reported_as_a_value_error(loader, tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        loader(str(path))


@pytest.mark.parametrize("loader", [load_weights, load_calibration])
def test_a_directory_is_not_mistaken_for_a_weights_file(loader, tmp_path):
    # Path.is_file() is False for a directory, so this must fall through to the registry error
    with pytest.raises(ValueError, match="Could not find"):
        loader(str(tmp_path))


@pytest.mark.parametrize("loader", [load_weights, load_calibration])
def test_missing_file_is_reported_as_a_value_error(loader, tmp_path):
    with pytest.raises(ValueError, match="Could not find"):
        loader(str(tmp_path / "absent.json"))


@pytest.mark.parametrize("loader", [load_weights, load_calibration])
def test_empty_identifier_does_not_crash(loader):
    # "" is a falsy path; Path("").is_file() is False, so the registry error is the contract
    with pytest.raises(ValueError, match="Could not find"):
        loader("")
