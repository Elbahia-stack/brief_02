import pytest
import pandas as pd
from piline import charg_data, qualite_data, encoder_d, scale_numeric, modelisation

@pytest.fixture
def df():
    return pd.read_csv("data_brief_2.csv")

def test_data_format_and_shape(df,target_col):


    X = df.drop(columns=[target_col])
    y = df[target_col]
    assert X.shape[0] == y.shape[0], "Le nombre de lignes de X et y ne correspond pas"


def test_mae(df):
    target_col = 'Delivery_Time_min'

    X = df.drop(columns=[target_col])
    y = df[target_col]

    results = modelisation(None, X, y, X, y)

    seuil = 10
    mae_rf = results["RandomForest"]["MAE"]
    mae_svr = results["SVR"]["MAE"]

    # Test : vérifier que les MAE ne dépassent pas le seuil
    assert mae_rf <= seuil, f"MAE RandomForest trop élevée : {mae_rf:.2f} > {seuil}"
    assert mae_svr <= seuil, f" MAE SVR trop élevée : {mae_svr:.2f} > {seuil}"