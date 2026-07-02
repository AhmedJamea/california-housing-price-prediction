from sklearn.model_selection import GridSearchCV
import numpy as np

def grid_search(
    model,
    features,
    labels,
    parameter_grid,
    cv=5
):
    search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        return_train_score=True
    )

    search.fit(features, labels)

    return search


def display_grid_search_results(search):

    results = search.cv_results_

    for mean_score, parameters in zip(
        results["mean_test_score"],
        results["params"]
    ):
        rmse = np.sqrt(-mean_score)

        print(f"{rmse:.2f} -> {parameters}")


def print_best_model(search):
    print("\nBest Parameters:")
    print(search.best_params_)

    print("\nBest Estimator:")
    print(search.best_estimator_)