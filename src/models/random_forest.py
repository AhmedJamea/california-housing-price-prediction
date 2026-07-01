from sklearn.ensemble import RandomForestRegressor


def train_random_forest(features, labels):
    model = RandomForestRegressor(random_state=42)

    model.fit(features, labels)

    return model


def predict(model, features):
    return model.predict(features)