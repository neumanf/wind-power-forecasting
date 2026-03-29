from sklearn.ensemble import RandomForestRegressor

def train_model(X_train, y_train, params):
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    return model
