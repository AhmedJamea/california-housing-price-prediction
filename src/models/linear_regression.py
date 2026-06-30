from sklearn.linear_model import LinearRegression

def train_linear_regression(featuers, labels):
    model = LinearRegression()
    model.fit(featuers, labels)
    return model

def predict(model, features):
    return model.predict(features)