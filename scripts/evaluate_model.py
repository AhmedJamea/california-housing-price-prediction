from src.get_data import load_housing_data
from src.split_data import stratified_split
from src.prepare_data import separate_features_and_labels

from src.preprocessing.pipeline import build_full_pipeline

from src.models.factory import create_model
from src.models.evaluation import (
    evaluate_model,
    display_scores
)

housing = load_housing_data()

train_set, _ = stratified_split(housing)

features, labels = separate_features_and_labels(train_set)

pipeline = build_full_pipeline(features)

prepared_features = pipeline.fit_transform(features)

models = [
    "linear_regression",
    "decision_tree",
    "random_forest",
    "support_vector_regression"
]

for model_name in models:

    print("=" * 50)
    print(model_name)

    model = create_model(model_name)

    scores = evaluate_model(
        model,
        prepared_features,
        labels
    )

    display_scores(scores)