from src.get_data import load_housing_data
from src.split_data import stratified_split
from src.prepare_data import separate_features_and_labels

from src.preprocessing.pipeline import build_full_pipeline

from src.models.factory import create_model
from src.models.trainer import train_model

housing = load_housing_data()

train_set, test_set = stratified_split(housing)

features, labels = separate_features_and_labels(train_set)

pipeline = build_full_pipeline(features)

prepared_features = pipeline.fit_transform(features)

model = create_model("random_forest")

trained_model = train_model(
    model,
    prepared_features,
    labels
)

print("Training completed.")