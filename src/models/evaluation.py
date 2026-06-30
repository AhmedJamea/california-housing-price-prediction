import numpy as np
from sklearn.metrics import mean_squared_error

def calculate_rmse(lables, predictions):

    mse = mean_squared_error(
        lables,
        predictions
    )

    return np.sqrt(mse)

