import pytest
import pandas as pd
from piline import charg_data, qualite_data, encoder_d, scale_numeric, modelisation



def test_data_format_and_shape(charg_data):
   assert charg_data.shape==(1000,9)


def test_mae(charg_data):
    target_col = 'Delivery_Time_min'
    X = charg_data.drop(columns=[target_col])
    y = charg_data[target_col]
    results = modelisation(None, X, y, X, y)
    seuil = 10
    mae_rf = results["RandomForest"]["MAE"]
    mae_svr = results["SVR"]["MAE"]
    assert mae_rf <= seuil, f"MAE RandomForest trop élevée : {mae_rf:.2f} > {seuil}"
    assert mae_svr <= seuil, f" MAE SVR trop élevée : {mae_svr:.2f} > {seuil}"