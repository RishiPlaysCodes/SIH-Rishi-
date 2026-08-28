"""Config loading, YAML subset parser round-trips, and overrides."""

from sentinelx.config import DEFAULT_CONFIG, deep_merge, dump_yaml, load_config, parse_yaml


def test_default_config_loads():
    cfg = load_config()
    assert cfg.seed == 1337
    assert cfg.get("experiment.model_type") == "linear_transition"
    assert cfg.get("deviation.weights.feature") == 0.5


def test_yaml_subset_roundtrip():
    text = dump_yaml(DEFAULT_CONFIG)
    parsed = parse_yaml(text)
    assert parsed["experiment"]["seed"] == 1337
    assert parsed["data"]["num_hosts"] == 18
    assert parsed["deviation"]["anomaly_threshold"] == 0.45
    # nested dict + typed scalars survive
    assert isinstance(parsed["graph"]["directed"], bool)


def test_yaml_parses_types_and_comments():
    text = """
    # a comment
    experiment:
      name: demo        # inline comment
      seed: 42
      ratio: 0.25
      enabled: true
      note: null
    list_section:
      - 1
      - 2
      - 3
    """
    parsed = parse_yaml(text)
    assert parsed["experiment"]["name"] == "demo"
    assert parsed["experiment"]["seed"] == 42
    assert parsed["experiment"]["ratio"] == 0.25
    assert parsed["experiment"]["enabled"] is True
    assert parsed["experiment"]["note"] is None
    assert parsed["list_section"] == [1, 2, 3]


def test_deep_merge_and_overrides():
    cfg = load_config(overrides={"experiment": {"seed": 7, "model_type": "ewma"}})
    assert cfg.seed == 7
    assert cfg.get("experiment.model_type") == "ewma"
    # untouched keys remain
    assert cfg.get("data.num_hosts") == 18


def test_deep_merge_is_nondestructive():
    base = {"a": {"x": 1, "y": 2}}
    merged = deep_merge(base, {"a": {"y": 20, "z": 3}})
    assert merged == {"a": {"x": 1, "y": 20, "z": 3}}
    assert base == {"a": {"x": 1, "y": 2}}  # original untouched
