import pandas as pd
from sklearn.impute import SimpleImputer

def separate_features_and_labels(train_set):

    housing = train_set.drop(
        "median_house_value",
        axis=1
    )

    housing_labels = train_set[
        "median_house_value"
    ].copy()

    return housing, housing_labels


