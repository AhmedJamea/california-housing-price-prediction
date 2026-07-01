from sklearn.model_selection import cross_val_score
import numpy as np


def evaluate_model(model, X_train, y_train, cv=10):
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        scoring="neg_mean_squared_error",
        cv=cv
    )

    rmse_scores = np.sqrt(-scores)

    return rmse_scores


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())