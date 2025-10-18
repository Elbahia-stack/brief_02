import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error,r2_score

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
def charg_data():

    return pd.read_csv("data_brief_2.csv")


def qualite_data(p):
    df=p.copy()
    df.fillna({
        'Weather': df['Weather'].mode()[0],
        'Traffic_Level': df['Traffic_Level'].mode()[0],
        'Time_of_Day': df['Time_of_Day'].mode()[0],
        'Courier_Experience_yrs': df['Courier_Experience_yrs'].median()
    }, inplace=True)
    df.isnull().sum()
    df.duplicated().sum()
    return df
def encoder_d(df):
    c_cols = df.select_dtypes(include='object').columns
    if len(c_cols) == 0:
        return df
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df[c_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(c_cols))

    df = pd.concat([df, encoded_df], axis=1)
    df = df.drop(columns=c_cols)

    return df

def scale_numeric(df):
    features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    features.remove('Delivery_Time_min')
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df
def best_features(df):
    X = df.drop('Delivery_Time_min', axis=1)
    y = df['Delivery_Time_min']
    selector = SelectKBest(score_func=f_regression, k=5)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    X_selected = pd.DataFrame(X_new, columns=selected_features)
    X= df[X_selected.tolist()]
    return X,y
def split(X,y):



    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test

def modelisation( X_train, y_train, X_test, y_test):
    rf = RandomForestRegressor(random_state=42)
    svr = SVR()

    # Grilles d’hyperparamètres
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10]
    }

    param_grid_svr = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    # Optimisation par validation croisée
    grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='neg_mean_absolute_error', n_jobs=1)
    grid_svr = GridSearchCV(svr, param_grid_svr, cv=5, scoring='neg_mean_absolute_error', n_jobs=1)

    # Entraînement
    grid_rf.fit(X_train, y_train)
    grid_svr.fit(X_train, y_train)

    # Prédictions
    y_pred_rf = grid_rf.best_estimator_.predict(X_test)
    y_pred_svr = grid_svr.best_estimator_.predict(X_test)

    # 🔹 Évaluation
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    mae_svr = mean_absolute_error(y_test, y_pred_svr)
    r2_svr = r2_score(y_test, y_pred_svr)


    results = {
        "RandomForest": {"MAE": mae_rf, "R2": r2_rf},
        "SVR": {"MAE": mae_svr, "R2": r2_svr}
    }

    return results