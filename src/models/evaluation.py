from sklearn.model_selection import cross_val_score
import numpy as np


def evaluate_model(model, features, lables, cv=10):
    scores = cross_val_score(
        model,
        features,
        lables,
        scoring="neg_mean_squared_error",
        cv=cv
    )

    rmse_scores = np.sqrt(-scores)

    return rmse_scores


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())