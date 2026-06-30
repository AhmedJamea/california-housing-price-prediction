from src.get_data import load_housing_data
from src.split_data import stratified_split

from src.prepare_data import (
    separate_features_and_labels,
    fit_numerical_imputer,
    transform_numerical_data,
)

from src.categorical_processing import (
    get_categorical_features,
    prepare_categorical_data,
)

from src.transformers import CombinedAttributesAdder


housing = load_housing_data()

train_set, test_set = stratified_split(housing)

housing_features, housing_labels = separate_features_and_labels(train_set)

imputer = fit_numerical_imputer(housing_features)

housing_num = transform_numerical_data(
    housing_features,
    imputer
)

housing_cat = get_categorical_features(
    housing_features
)

housing_cat, encoder = prepare_categorical_data(
    housing_cat
)

print("Numerical data shape:", housing_num.shape)
print("Categorical data shape:", housing_cat.shape)

print("\nCategories:")
print(encoder.categories_)

print("\nTraining labels:")
print(housing_labels.head())