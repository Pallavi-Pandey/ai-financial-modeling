import config


def test_model_params_has_both_models():
    assert "xgboost" in config.MODEL_PARAMS
    assert "random_forest" in config.MODEL_PARAMS


def test_xgboost_params_structure():
    xgb = config.MODEL_PARAMS["xgboost"]
    assert xgb["n_estimators"] == 100
    assert xgb["max_depth"] == 4
    assert xgb["learning_rate"] == 0.1


def test_random_forest_params_structure():
    rf = config.MODEL_PARAMS["random_forest"]
    assert rf["n_estimators"] == 100
    assert rf["max_depth"] == 6


def test_forecast_periods_is_six():
    assert config.FORECAST_PERIODS == 6


def test_test_size_is_point_two():
    assert config.TEST_SIZE == 0.2


def test_data_path_is_string():
    assert isinstance(config.DATA_PATH, str)
    assert config.DATA_PATH.endswith(".csv")
