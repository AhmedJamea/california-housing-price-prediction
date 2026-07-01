from sklearn.tree import DecisionTreeRegressor

def train_decision_tree(features, lables):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(features, lables)
    return model

def predict(model, features):
    return model.predict(features)
