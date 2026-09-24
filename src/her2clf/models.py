from sklearn.linear_model import LogisticRegression

def train_logistic_regression(X_train,y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train,y_train)

    return model

from xgboost import XGBClassifier

def train_xgboost(X_train, y_train):
    model= XGBClassifier(
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    return model