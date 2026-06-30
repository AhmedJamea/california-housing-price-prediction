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


def fit_numerical_imputer(housing: pd.DataFrame):
    """
    Learn the median of every numerical column
    """

    # remove the categorical column
    housing_num = housing.drop("ocean_proximity", axis=1)

    # craete the imputer
    imputer = SimpleImputer(strategy="median")

    # learn the medians
    imputer.fit(housing_num)

    return imputer


def transform_numerical_data(
    housing: pd.DataFrame,
    imputer: SimpleImputer
):
    """
    Replace missing numerical values using
    the medians learned by the fitted imputer.
    """

    # remove the categorical column
    housing_num = housing.drop("ocean_proximity", axis=1)

    # replace the missing values
    housing_num_prepared = pd.DataFrame(
        imputer.transform(housing_num),
        columns=housing_num.columns,
        index=housing_num.index
    )

    return housing_num_prepared