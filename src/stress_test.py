def stress_test_amount_shift(X, model, factor):
    X_shifted = X.copy()
    X_shifted["Amount"] *= factor
    return model.predict_proba(X_shifted)[:, 1].mean()


def stress_test_time_shift(X, model):
    X_shifted = X.copy()
    X_shifted["Hour"] = (X_shifted["Hour"] + 6) % 24
    return model.predict_proba(X_shifted)[:, 1].mean()
