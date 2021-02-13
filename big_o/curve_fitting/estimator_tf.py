import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import MeanAbsolutePercentageError


def build_model(dim):
    model = Sequential(
        [
            Dense(
                1,
                kernel_initializer="he_uniform",
                kernel_constraint=NonNeg(),
                input_shape=(dim,),
            )
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    return model


def fit_nn(X, seed=111):
    X = X.copy().sample(frac=1, random_state=seed)

    preds = []
    losses = []
    for col in X.columns:
        if col not in ["n", "sqrt", "logn"]:
            continue

        model = build_model(1)

        history = model.fit(X[[col]], X["mean"], batch_size=1024, epochs=1)
        losses.append(history.history.get("loss")[-1])
        y = model.predict(X[[col]].values)
        X[f"y-{col}"] = y
        preds.append(f"y-{col}")
    return X, preds, losses
