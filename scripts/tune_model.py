from src.get_data import load_housing_data
from src.split_data import stratified_split
from src.prepare_data import separate_features_and_labels

from src.preprocessing.pipeline import (
    build_full_pipeline,
)

from src.models.factory import create_model
from src.models.tuning import (
    grid_search,
    print_best_model,
    display_grid_search_results,
)

# Load dataset
housing = load_housing_data()

# Split dataset
train_set, _ = stratified_split(housing)

# Features and labels
housing_features, housing_labels = (
    separate_features_and_labels(train_set)
)

# Preprocess
pipeline = build_full_pipeline(
    housing_features
)

housing_prepared = pipeline.fit_transform(
    housing_features
)

# Model
model = create_model("random_forest")

# Hyperparameter search space
parameter_grid = [
    {
        "n_estimators": [3, 10, 30],
        "max_features": [2, 4, 6, 8],
    },
    {
        "bootstrap": [False],
        "n_estimators": [3, 10],
        "max_features": [2, 3, 4],
    },
]

search = grid_search(
    model=model,
    parameter_grid=parameter_grid,
    features=housing_prepared,
    labels=housing_labels,
)

print_best_model(search)

print()

display_grid_search_results(search)