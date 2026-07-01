from src.get_data import load_housing_data
from src.split_data import stratified_split
from src.prepare_data import separate_features_and_labels

from src.preprocessing.pipeline import (
    build_full_pipeline
)

from src.models.factory import create_model
from src.models.trainer import train_model, predict

from src.models.evaluation import (
    evaluate_model,
    display_scores
)

# Load dataset
housing = load_housing_data()

# Split into train/test
train_set, test_set = stratified_split(housing)

# Separate features and labels
housing_features, housing_labels = separate_features_and_labels(train_set)

# Build preprocessing pipeline
preprocessing_pipeline = build_full_pipeline(
    housing_features
)

# Prepare training features
housing_prepared = preprocessing_pipeline.fit_transform(
    housing_features
)

# choose model
model = create_model("random_forest")

# train model
model = train_model(
    model,
    housing_prepared,
    housing_labels
)

# predict on training set
predictions = predict(
    model,
    housing_prepared
)

# evaluate 
scores = evaluate_model(
    model,
    housing_prepared,
    housing_labels
)

display_scores(scores)