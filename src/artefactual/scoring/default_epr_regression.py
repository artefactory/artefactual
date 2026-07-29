from sklearn.linear_model import LogisticRegression


class DefaultEprLogisticRegression(LogisticRegression):
    def __sklearn_is_fitted__(self) -> bool:
        return True
