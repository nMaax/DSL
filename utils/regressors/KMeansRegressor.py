import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression


class KMeansRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_clusters=3, base_regressor=None):
        """
        A regression model that combines K-means clustering with Linear Regression.
        """
        self.n_clusters = n_clusters
        self.base_regressor = base_regressor if base_regressor else LinearRegression()
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.regressors_ = {}

    def fit(self, X, y):
        """
        Fit K-means and cluster-specific regressors.
        """
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        # Fit K-means to find clusters
        self.kmeans_.fit(X)
        labels = self.kmeans_.predict(X)

        # Fit a separate regressor for each cluster
        for cluster in range(self.n_clusters):
            mask = (labels == cluster)
            regressor = clone(self.base_regressor)
            regressor.fit(X[mask], y[mask])
            self.regressors_[cluster] = regressor

        return self

    def predict(self, X):
        """
        Predict target values using cluster-specific regressors.
        """
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        # Predict cluster labels for input data
        labels = self.kmeans_.predict(X)

        # Predict using the corresponding regressor for each cluster
        predictions = np.zeros(X.shape[0])
        for cluster in range(self.n_clusters):
            mask = (labels == cluster)
            predictions[mask] = self.regressors_[cluster].predict(X[mask])

        return predictions


if __name__ == "__main__":
    X = np.random.rand(200, 5)  # 200 samples, 5 features
    y = 3 * X[:, 0] + 2 * X[:, 1] - 4 * X[:, 2] + np.random.randn(200)  # Non-linear relationship

    model = KMeansRegressor(n_clusters=3)

    model.fit(X, y)

    y_pred = model.predict(X)

    print("Predictions:", y_pred)
