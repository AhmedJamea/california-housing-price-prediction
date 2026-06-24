from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np


def create_income_category(housing):
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0., 1.5, 3., 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    return housing


def stratified_split(housing):

    housing = create_income_category(housing)

    split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    for train_idx, test_idx in split.split(
        housing,
        housing["income_cat"]
    ):
        train_set = housing.loc[train_idx]
        test_set = housing.loc[test_idx]

    return train_set, test_set