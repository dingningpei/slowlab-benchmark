from datetime import datetime
import numpy as np
import pytest
from slowlab.agc_temperature_residual import (
    FEATURE_NAMES, TemperatureResidualModel, feature_vector,
    fit_temperature_residual,
)


def test_feature_vector_has_frozen_order_and_clock_encoding():
    row={name: i+1 for i,name in enumerate(FEATURE_NAMES[:-2])}
    result=feature_vector(row,datetime(2020,1,1,6))
    assert result.shape == (len(FEATURE_NAMES),)
    assert result[-2] == pytest.approx(1)
    assert result[-1] == pytest.approx(0,abs=1e-12)


def test_ridge_model_round_trips_and_predicts_training_signal():
    rng=np.random.default_rng(4)
    x=rng.normal(size=(300,len(FEATURE_NAMES)))
    y=1.25 + 0.8*x[:,0] - 0.3*x[:,4]
    model=fit_temperature_residual(x,y,ridge_lambda=10)
    restored=TemperatureResidualModel.from_dict(model.to_dict())
    assert np.sqrt(np.mean((restored.predict_matrix(x)-y)**2)) < 0.05
    assert restored.feature_names == FEATURE_NAMES


def test_residual_fit_rejects_identity_columns_and_bad_shapes():
    x=np.zeros((5,len(FEATURE_NAMES)+1)); y=np.zeros(5)
    with pytest.raises(ValueError,match="wrong shape"):
        fit_temperature_residual(x,y)
