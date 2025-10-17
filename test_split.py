import pytest
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

@pytest.fixture
def df():
    return pd.read_csv("data_brief_2.csv")

def test_data_format_and_shape(df):

    assert isinstance(df, pd.DataFrame), "La donnée importée n'est pas un DataFrame pandas"
    target_col = 'Delivery_Time_min'
    assert target_col in df.columns, f"La colonne cible {target_col} est manquante"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    assert X.shape[0] == y.shape[0], "Le nombre de lignes de X et y ne correspond pas"


def test_mae(df):
    target_col = 'Delivery_Time_min'
    X = df.drop(columns=[target_col])
    y = df[target_col]


    c_cols = df.select_dtypes(include=['object']).columns


    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(X[c_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(c_cols))

    # Fusionner et supprimer les anciennes colonnes
    X = pd.concat([X.drop(columns=c_cols).reset_index(drop=True), encoded_df], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Définir modèles et paramètres
    rf = RandomForestRegressor(random_state=42)
    svr = SVR()

    param_grid_rf = {
        'n_estimators': [50, 100],
        'max_depth': [None, 5]
    }

    param_grid_svr = {
        'C': [0.1, 1],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    # GridSearchCV
    grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring='neg_mean_absolute_error', n_jobs=1)
    grid_svr = GridSearchCV(svr, param_grid_svr, cv=3, scoring='neg_mean_absolute_error', n_jobs=1)


    grid_rf.fit(X_train, y_train)
    grid_svr.fit(X_train, y_train)

    # Prédictions et MAE
    mae_rf = mean_absolute_error(y_test, grid_rf.best_estimator_.predict(X_test))
    mae_svr = mean_absolute_error(y_test, grid_svr.best_estimator_.predict(X_test))

    # Seuil
    seuil = 10

    assert mae_rf <= seuil, f"MAE RandomForest trop élevée: {mae_rf} > {seuil}"
    assert mae_svr <= seuil, f"MAE SVR trop élevée: {mae_svr} > {seuil}"