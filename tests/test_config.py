from cape_wm.config import load_config


def test_phase_c_inherits_phase_b_planner_configuration():
    config = load_config("configs/cape_phase_c.yaml")
    assert config["experiment"]["name"] == "cape_phase_c"
    assert config["planner"]["durations"] == [5, 10, 20, 40]
    assert "ablations" in config


def test_single_runtime_config_contains_portable_defaults():
    config = load_config("configs/runtime.yaml")
    assert config["runtime"]["device"] == "auto"
    assert config["runtime"]["num_workers"] == 0
    assert config["training"]["macro_batch_size"] > 0


def test_cc_rwkv_tworoom_config_has_unambiguous_horizon_units():
    config = load_config("configs/cc_rwkv/tworoom.yaml")
    protocol = config["protocol"]
    assert protocol["primitive_horizons"] == [5, 10, 20, 50, 100]
    assert protocol["action_block"] == protocol["observation_stride"] == 5
    assert [item // protocol["action_block"] for item in protocol["primitive_horizons"]] == [
        1,
        2,
        4,
        10,
        20,
    ]
    assert config["evaluation"]["no_intermediate_observations"] is True
    assert config["evaluation"]["controller"] == "none"
