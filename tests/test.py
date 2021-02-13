from big_o.timer import timeit
from big_o.visualize import lineplot, PlotData
from big_o.curve_fitting.estimator import ridge_fit
from big_o.curve_fitting.features import polynominal_features, load_dataset, Records


def divisors(number):
    """expected O(N^1/2)"""
    lower_divisors = []
    l_div_a = lower_divisors.append
    upper_divisors = []
    u_div_a = upper_divisors.append
    factor = 1
    while factor * factor <= number:
        if number % factor == 0:
            l_div_a(factor)
            if factor != number // factor:
                u_div_a(number // factor)
        factor += 1
    return lower_divisors + upper_divisors[::-1]


if __name__ == "__main__":
    saveimg = "tests/result.png"
    iter_size = 50
    max_power = 8
    split = 30
    # we divide each scale region [10^k, 10^(k+1)) by <split>, maximum k is <max_power>

    """
    The ideal data size equals or is greater than returned data size.
    That is because window size is rounded down. that size determine second loop depth, 
    lower window, deeper loop.

    For example:
    >>> split = 20
    >>> dx = (100 - 10) // split
    # dx is 4 but 4 * split = 80. So, additional two iteration is processed.
    """

    timerit = timeit(iter_size)
    ns = []
    means = []
    stds = []
    data_size = max_power * split
    for power in range(1, max_power + 1):
        identity = int("1" + "0" * power)
        dx = (identity * 9) // split
        for n in range(identity, 10 * identity, dx):
            mean, std = timerit(divisors, n)
            ns.append(n)
            means.append(mean)
            stds.append(std)

    assert len(ns) >= data_size, f"Few data as expected: {len(ns) - data_size}"

    records = Records(n=ns, mean=means, std=stds)

    df = load_dataset(records)
    X = polynominal_features(df)

    df_res, preds_rr, losses = ridge_fit(X, seed=111)
    best = preds_rr[losses.index(min(losses))]
    lineplot(
        PlotData(x=ns, y=means, legend="ref", err_interval=stds),
        PlotData(x=df_res["n"].tolist(), y=df_res[best].tolist(), legend=best),
        path=saveimg,
    )

    print(f"\nPassed function is predicted O({best[2:]})")
    print(
        "NOTE: This is an experimental feature. Make sure that the fitting curve is reliable with your eyes."
    )
    print(f"The fitting curve was saved in {saveimg}")
