import pytest

from src.services.common import config as c
from src.services.common.config import Config, ConfigError, ModelConfig, VectorConfig


def test_defaults_without_vectors():
    w = c.resolve_weights(Config(), vector_store_empty=True)
    assert w == pytest.approx({c.REGEX: 0.1, c.MODEL: 0.9, c.VECTOR: 0.0})


def test_defaults_switch_when_vector_store_has_data():
    cfg = Config(vector_catcher=True)
    assert c.resolve_weights(cfg, vector_store_empty=True) == pytest.approx(
        {c.REGEX: 0.1, c.MODEL: 0.9, c.VECTOR: 0.0}
    )
    assert c.resolve_weights(cfg, vector_store_empty=False) == pytest.approx(
        {c.REGEX: 0.1, c.MODEL: 0.65, c.VECTOR: 0.25}
    )


def test_defaults_rescale_when_a_layer_is_off():
    w = c.resolve_weights(Config(model_catcher=False), vector_store_empty=True)
    assert w == pytest.approx({c.REGEX: 1.0, c.MODEL: 0.0, c.VECTOR: 0.0})
    w = c.resolve_weights(
        Config(regex_catcher=False, vector_catcher=True), vector_store_empty=False
    )
    assert w[c.REGEX] == 0 and sum(w.values()) == pytest.approx(1.0)
    assert w[c.MODEL] / w[c.VECTOR] == pytest.approx(0.65 / 0.25)


def test_explicit_weights_are_used_as_given():
    cfg = Config(
        vector_catcher=True,
        detector_scoring={c.REGEX: 0.2, c.MODEL: 0.5, c.VECTOR: 0.3},
    )
    assert c.resolve_weights(cfg, vector_store_empty=True)[c.VECTOR] == 0.3


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"detector_scoring": {c.REGEX: 0.2, c.MODEL: 0.9}}, "sum to 1"),
        ({"detector_scoring": {c.REGEX: 0.1, c.MODEL: 0.8, c.VECTOR: 0.1}}, "switched off"),
        ({"detector_scoring": {c.REGEX: 0.0, c.MODEL: 1.0}}, "switched on"),
        ({"detector_scoring": {"regex": 1.0}}, "unknown"),
        ({"detector_scoring": {c.REGEX: -0.1, c.MODEL: 1.1}}, "between 0 and 1"),
        ({"regex_catcher": False, "model_catcher": False}, "at least one"),
        ({"detect_threshold": 1.5}, "between 0 and 1"),
    ],
)
def test_invalid_configs_raise(kwargs, message):
    with pytest.raises(ConfigError, match=message):
        Config(**kwargs)


def test_missing_keys_count_as_zero():
    cfg = Config(detector_scoring={c.REGEX: 0.1, c.MODEL: 0.9})
    assert cfg.detector_scoring == {c.REGEX: 0.1, c.MODEL: 0.9, c.VECTOR: 0.0}


def test_vector_config_validation_and_env(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "https://example.invalid")
    monkeypatch.setenv("QDRANT_API_KEY", "secret-key")
    vc = VectorConfig()
    assert (vc.catch_threshold, vc.learn_threshold, vc.top_k) == (0.8, 0.7, 10)
    assert vc.provider_url == "https://example.invalid"
    assert "secret-key" not in repr(vc)
    with pytest.raises(ConfigError):
        VectorConfig(catch_threshold=2)
    with pytest.raises(ConfigError):
        VectorConfig(top_k=0)


def test_model_config_defaults_and_secrets(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.delenv("HAVEL_HOSTED_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or_secret")
    mc = ModelConfig()
    assert mc.repo_id == "Gigadelux/havel_base" and mc.precision == "int8"
    assert mc.provider_api_key == "or_secret"
    assert "hf_secret" not in repr(mc) and "or_secret" not in repr(mc)
    with pytest.raises(ConfigError):
        ModelConfig(precision="int4")
    with pytest.raises(ConfigError):
        ModelConfig(min_chunk_tokens=600)
