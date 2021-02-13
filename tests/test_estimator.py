from math import sqrt
from random import random
import plotly.graph_objects as go

from big_o.curve_fitting.estimator import ridge_fit

# from big_o.curve_fitting.estimator_tf import fit_nn
from big_o.curve_fitting.features import polynominal_features, load_dataset, Records


def show(df, columns=None):
    df = df.sort_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.n, y=df["mean"], mode="lines+markers", name="ref"))
    if columns:
        if not isinstance(columns, list):
            columns = [columns]
        for col in columns:
            fig.add_trace(go.Scatter(x=df.n, y=df[col], mode="lines+markers", name=col))
    fig.show()


def test_estimator():
    records = Records(
        n=list(range(1, 101)),
        mean=[sqrt(f) + random() for f in range(1, 101)],
        std=[random() for _ in range(1, 101)],
    )

    df = load_dataset(records)
    X = polynominal_features(df)

    df_res, preds_rr, losses = ridge_fit(X, seed=111)
    best = preds_rr[losses.index(min(losses))]
    # show(df_res, columns=best)
    assert best == "y-sqrt"
