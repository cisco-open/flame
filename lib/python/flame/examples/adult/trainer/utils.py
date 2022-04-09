"""utils for adult dataset."""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


class MyAdultDataset(torch.utils.data.Dataset):
    """MyDataset class."""

    def __init__(self, X: np.array, y=None, features=None):
        """
        Initialize.

        Parameters:
            X: numpy array
            y: optional target column
            features: optional feature list
        """
        self.X = X
        self.y = y

        self.features = features

    def __len__(self):
        """Return the length of dataset."""
        return self.X.shape[0]

    def __getitem__(self, idx):
        """Get item."""
        if not self.features:
            if self.y is not None:
                return self.X[idx, :], self.y[idx]
            else:
                return self.X[idx, :]

        X = []

        def onehot(x, n):
            res = [0] * n
            res[int(x)] = 1
            return res

        for i, f in enumerate(self.features):
            if f.categorical:
                X.extend(onehot(self.X[idx, i], len(f.values)))
            else:
                X.append(self.X[idx, i])

        if self.y is not None:
            return np.array(X, dtype="float32"), self.y[idx].astype("float32")
        else:
            return np.array(X, dtype="float32")


class Feature:
    """Feature class."""

    def __init__(self,
                 name,
                 dtype,
                 description,
                 categorical=False,
                 values=None) -> None:
        """Initialize."""
        self.name = name
        self.dtype = dtype
        self.description = description
        self.categorical = categorical
        self.values = values

    def __repr__(self) -> str:
        """Represent."""
        return f"{self.name}:{self.dtype}"


def clean_dataframe(df: pd.DataFrame, clear_nans=True, extra_symbols="?"):
    """Clean dataframe by removing NaNs."""
    if not clear_nans:
        return

    for i in df:
        df[i].replace('nan', np.nan, inplace=True)

        for s in extra_symbols:
            df[i].replace(s, np.nan, inplace=True)

    df.dropna(inplace=True)


def process_dataframe(df: pd.DataFrame,
                      target_column=None,
                      normalize="Scalar"):
    """Process dataframe and return numpy datasets and feature info."""
    if normalize and normalize == "Scalar":
        num_d = df.select_dtypes(exclude=['object', 'category'])
        df[num_d.columns] = StandardScaler().fit_transform(num_d)

    y = None
    if target_column:
        y = df.pop(target_column)
        if y.dtype in ("object", "category"):
            y = y.factorize(sort=True)[0]

    features = []

    for c in df:
        if df.dtypes[c] == "object":
            fact = df[c].factorize(sort=True)
            df[c] = fact[0]

            values = {i: v for i, v in enumerate(fact[1])}
            f = Feature(c, "integer", c, categorical=True, values=values)
        else:
            f = Feature(c, "float32", c)
        features.append(f)

    X = df.to_numpy().astype('float32')

    return X, y, features
