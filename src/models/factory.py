from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def create_model(model_name):

    models = {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(random_state=42),
        "random_forest": RandomForestRegressor(random_state=42),
        "support_vector_regression": SVR(
            kernel="rbf",
            C=100,
            gamma="scale",
            epsilon=0.1
        ),
    }

    return models[model_name]