import pytest
import pandas as pd
from piline import  qualite_data, encoder_d, scale_numeric, modelisation
@pytest.fixture
def data():

    return pd.read_csv("data_brief_2.csv")

def test_data_format_and_shape(data):
   assert data.shape==(1000,9)


def test_mae(data):
    df = qualite_data(data)
    df = encoder_d(df)
    df = scale_numeric(df)
    target_col = 'Delivery_Time_min'
    X = data.drop(columns=[target_col])
    y = data[target_col]
    results = modelisation(None, X, y, X, y)
    seuil = 10
    mae_rf = results["RandomForest"]["MAE"]
    mae_svr = results["SVR"]["MAE"]
    assert mae_rf <= seuil, f"MAE RandomForest trop élevée : {mae_rf:.2f} > {seuil}"
    assert mae_svr <= seuil, f" MAE SVR trop élevée : {mae_svr:.2f} > {seuil}"