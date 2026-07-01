from sklearn.linear_model import LinearRegression

def train_linear_regression(features, labels):
    model = LinearRegression()
    model.fit(features, labels)
    return model

def predict(model, features):
    return model.predict(features)