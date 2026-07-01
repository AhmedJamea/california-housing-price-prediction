def train_model(model, features, labels):
    model.fit(features, labels)
    return model


def predict(model, features):
    return model.predict(features)