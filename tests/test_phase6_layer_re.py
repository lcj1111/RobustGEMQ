import pickle

import pytest

from scripts.phase6.validate_layer_re import main


def test_layer_re_validator_accepts_complete_tensor(tmp_path, monkeypatch):
    path = tmp_path / "layer-re.pkl"
    output = tmp_path / "summary.json"
    values = {layer: {expert: {1: 1.0, 2: 0.5, 3: 0.25} for expert in range(64)} for layer in range(16)}
    path.write_bytes(pickle.dumps(values))
    monkeypatch.setattr("sys.argv", ["validate", str(path), "--output", str(output)])
    main()
    assert output.is_file()


def test_layer_re_validator_rejects_missing_expert(tmp_path, monkeypatch):
    path = tmp_path / "layer-re.pkl"
    output = tmp_path / "summary.json"
    values = {layer: {expert: {1: 1.0, 2: 0.5, 3: 0.25} for expert in range(64)} for layer in range(16)}
    del values[0][63]
    path.write_bytes(pickle.dumps(values))
    monkeypatch.setattr("sys.argv", ["validate", str(path), "--output", str(output)])
    with pytest.raises(ValueError, match="experts 0..63"):
        main()
