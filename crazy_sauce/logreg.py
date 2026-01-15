from __future__ import annotations

"""
Minimal logistic regression with batch gradient descent and internal scaling.
Good enough for the assignments without pulling in heavier dependencies.
"""

import numpy as np


class LogisticRegressionGD:
    """
    Minimal logistic regression trained with batch gradient descent.
    Features are standardized internally for stability.
    """

    def __init__(
        self,
        lr: float = 0.1,
        max_iter: int = 5000,
        tol: float = 1e-6,
        l2: float = 0.0,
        verbose: bool = False,
    ):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.l2 = l2
        self.verbose = verbose
        self.weights_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.n_iter_: int = 0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _add_bias(X: np.ndarray) -> np.ndarray:
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0)
            self.std_[self.std_ == 0] = 1.0
        return (X - self.mean_) / self.std_

    def fit(self, X, y):
        """Train the model with batch gradient descent."""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        X_scaled = self._standardize(X_arr)
        X_bias = self._add_bias(X_scaled)
        self.weights_ = np.zeros(X_bias.shape[1])

        for step in range(1, self.max_iter + 1):
            preds = self._sigmoid(X_bias @ self.weights_)
            error = preds - y_arr
            grad = (X_bias.T @ error) / len(y_arr)
            if self.l2:
                grad[1:] += self.l2 * self.weights_[1:]

            new_weights = self.weights_ - self.lr * grad
            if np.linalg.norm(new_weights - self.weights_) < self.tol:
                self.weights_ = new_weights
                self.n_iter_ = step
                break

            self.weights_ = new_weights
            self.n_iter_ = step

            if self.verbose and step % 500 == 0:
                loss = -np.mean(
                    y_arr * np.log(preds + 1e-12)
                    + (1 - y_arr) * np.log(1 - preds + 1e-12)
                )
                print(f"[GD] step={step}, loss={loss:.4f}")

        self.intercept_ = float(self.weights_[0])
        self.coef_ = self.weights_[1:]
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Return P(y=1) for each row in X."""
        if self.weights_ is None:
            raise RuntimeError("Model is not fitted.")
        X_arr = np.asarray(X, dtype=float)
        X_scaled = self._standardize(X_arr)
        preds = self._sigmoid(self._add_bias(X_scaled) @ self.weights_)
        return preds

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """Binary predictions using the provided threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)
