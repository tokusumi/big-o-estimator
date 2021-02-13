from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def ridge_fit(X, seed=111):
    X = X.copy().sample(frac=1, random_state=seed)

    preds = []
    losses = []
    for col in X.columns:
        if col not in ["n", "sqrt", "logn"]:
            continue

        rr = Ridge(random_state=111)
        rr.fit(X[[col]], X["mean"])
        y = rr.predict(X[[col]])
        X[f"y-{col}"] = y
        losses.append(mean_squared_error(X["mean"], y))
        preds.append(f"y-{col}")
    return X, preds, losses
