import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import (
    check_array,
    check_X_y,
    column_or_1d
)
from sklearn.utils.validation import check_is_fitted
from speed_logit_reg._speed_logit_reg import speedLogisticRegression, logitLinkInv

class SpeedLogitReg(BaseEstimator, TransformerMixin):
    def __init__(self, fit_intercept=True, init_coef=None, max_iter=1000, tol=1e-4):
        self.fit_intercept = fit_intercept
        self.init_coef = init_coef
        self.max_iter = max_iter
        self.tol = tol
        self.fitted = False

    def fit(self, X, y, weights=None):
        X = check_array(X, force_all_finite=True, ensure_2d=True, dtype=np.float64)
        y = column_or_1d(y)
        y = check_array(y, ensure_2d=False, dtype=np.float64)
        X, y = check_X_y(X, y)

        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))

        if weights is None:
            weights = np.ones(X.shape[0], dtype=np.float64)

        if self.init_coef is None:
            self.init_coef = np.zeros(X.shape[1], dtype=np.float64)

        self._coef = speedLogisticRegression(X, y, weights, self.init_coef, self.tol, self.max_iter)
        self.fitted = True
        return self

    def predict(self, X):
        if self.fitted:
            X = check_array(X, force_all_finite=True, ensure_2d=True, dtype=np.float64)
            if self.fit_intercept:
                X = np.hstack((np.ones((X.shape[0], 1)), X))
            return np.matmul(X, self._coef)
        else:
            raise Exception('Model is not fitted yet.')

    def predict_proba(self, X):
        if self.fitted:
            return logitLinkInv(self.predict(X))
        else:
            raise Exception('Model is not fitted yet.')
