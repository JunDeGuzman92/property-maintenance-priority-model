import numpy as np


class LogisticPriorityModel:
    def __init__(self, learning_rate=0.08, epochs=2500, l2=0.02):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        xs = (x - self.mean_) / self.scale_
        n, p = xs.shape
        self.weights_ = np.zeros(p)
        self.bias_ = 0.0
        for _ in range(self.epochs):
            score = xs @ self.weights_ + self.bias_
            probs = self._sigmoid(score)
            error = probs - y
            self.weights_ -= self.learning_rate * ((xs.T @ error) / n + self.l2 * self.weights_)
            self.bias_ -= self.learning_rate * error.mean()
        return self

    def predict_proba(self, x):
        x = np.asarray(x, dtype=float)
        xs = (x - self.mean_) / self.scale_
        return self._sigmoid(xs @ self.weights_ + self.bias_)

    def predict(self, x, threshold=0.5):
        return (self.predict_proba(x) >= threshold).astype(int)

    def to_dict(self, feature_names, metrics):
        return {
            "model_type": "from_scratch_logistic_regression",
            "feature_names": list(feature_names),
            "weights": self.weights_.round(6).tolist(),
            "bias": round(float(self.bias_), 6),
            "mean": self.mean_.round(6).tolist(),
            "scale": self.scale_.round(6).tolist(),
            "metrics": metrics,
            "privacy_note": "Synthetic data only. Human review required for any operational use.",
        }

