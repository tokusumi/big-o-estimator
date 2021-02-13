from typing import List, Tuple, Optional
from dataclasses import dataclass

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class PlotData:
    x: List[Tuple[int, float]]
    y: List[float]
    legend: Optional[str] = None
    err_interval: Optional[float] = None

    def raise_if_invalid(self):
        if len(self.x) != len(self.y):
            raise ValueError("PlotData x and y attributes must be same size")


def lineplot(*data: PlotData, path=""):
    """save line plot for data
    Args:
        data: `PlotData` to plot in figure
        path: filename to save figure

    Examples:
    >>> from big_o.visualize import lineplot, PlotData
    >>> ns = [1, 2, 3]; yA = [1, 4, 9]; yB = [1, 8, 27]
    >>> lineplot(
    ...     PlotData(x=ns, y=yA, legend="A"),
    ...     PlotData(x=ns, y=yB, legend="B"),
    ...     path="example.png",
    ... )
    """

    @dataclass
    class PlotDatas:
        data: List[PlotData]

        def to_dict(self):
            items = {"n": [], "elapsed_time": [], "legend": []}
            for item in self.data:
                item.raise_if_invalid()
                items["n"] += item.x
                items["elapsed_time"] += item.y
                items["legend"] += [item.legend] * len(item.x)
            return items

    df = pd.DataFrame(PlotDatas(data=data).to_dict())
    line_plot = sns.lineplot(
        data=df,
        x="n",
        y="elapsed_time",
        hue="legend",
        markers=True,
    )
    line_plot.legend_.set_title(None)

    for d in data:
        if d.err_interval:
            y_np = np.array(d.y)
            err_np = np.array(d.err_interval)
            lower_bound = y_np - err_np
            upper_bound = y_np + err_np

            plt.fill_between(d.x, lower_bound, upper_bound, alpha=0.3)

    figure = line_plot.get_figure()
    figure.savefig(path)
