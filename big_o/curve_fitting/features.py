from typing import Tuple, List, Optional
from math import log2
from operator import mul
from functools import reduce
from dataclasses import dataclass


import pandas as pd
import numpy as np


@dataclass
class Records:
    n: List[int]
    mean: List[float]
    std: List[float]


def load_dataset(records: Records):
    records = {"n": records.n, "mean": records.mean, "std": records.std}
    df = pd.DataFrame(records)

    last_val = df["mean"][df["n"].argmax()]
    df = df[df["mean"] <= 2 * last_val]

    return df


@dataclass(frozen=True)
class BigOBase:
    column: str
    max_degree: int
    degree: int = 1


def get_index_map(ranges: List[BigOBase], start=0) -> Tuple[np.ndarray, int]:
    values = [f.max_degree + 1 for f in ranges]
    tot = reduce(mul, values)
    index_map = np.array(range(start, start + tot)).reshape(values)
    return index_map, tot


def indexer(ranges: List[BigOBase], start=0):
    index_map, size = get_index_map(ranges, start)

    def _index(target):
        arg = np.argwhere(index_map == target)
        if arg.shape[0] > 0:
            arg = arg[0].tolist()
            return [
                BigOBase(column=f.column, max_degree=f.max_degree, degree=i)
                for f, i in zip(ranges, arg)
            ]
        raise ValueError(f"{target} is out of range")

    return _index, size


def polynominal_features(df, ranges: Optional[List[BigOBase]] = None):
    if not ranges:
        ranges = [
            BigOBase(column="n", max_degree=3),
            BigOBase(column="sqrt", max_degree=1),
            BigOBase(column="logn", max_degree=3),
        ]
    indexer_, size = indexer(ranges)

    X = df[["mean", "n"]].assign(
        sqrt=lambda df: df["n"] ** (1 / 2),
        logn=lambda df: df["n"].apply(lambda x: log2(x)),
    )

    for i in range(size):
        indexs = indexer_(i)
        if sum([x.degree for x in indexs]) <= 1:
            continue
        _items = 1
        _name = ""
        for i in indexs:
            _items *= X[i.column] ** i.degree
        _name = "*".join([f"{i.column}{i.degree}" for i in indexs if i.degree != 0])
        X[_name] = _items

    return X
