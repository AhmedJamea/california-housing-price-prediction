import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def get_categorical_features(housing: pd.DataFrame):
    return housing[["ocean_proximity"]]

def prepare_categorical_data(housing_cat: pd.DataFrame):
    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

    housing_cat_prepared = encoder.fit_transform(housing_cat)

    return housing_cat_prepared, encoder