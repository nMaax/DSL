import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

class AgeGroupedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, 
                 age_threshold=40, 
                 classifier=None, 
                 young_regressor=None, 
                 old_regressor=None):
        self.age_threshold = age_threshold
        self.classifier = classifier if classifier else RandomForestClassifier(n_estimators=50, random_state=42)
        self.young_regressor = young_regressor if young_regressor else LinearRegression()
        self.old_regressor = old_regressor if old_regressor else LinearRegression()
        self.regressors_ = {}

    def fit(self, X, y):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        y_class = (y >= self.age_threshold).astype(int)

        self.classifier.fit(X, y_class)

        for label in np.unique(y_class):
            mask = (y_class == label)
            regressor = clone(self.young_regressor if label == 0 else self.old_regressor)
            regressor.fit(X[mask], y[mask])
            self.regressors_[label] = regressor
        return self

    def predict(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        y_class_pred = self.classifier.predict(X)

        predictions = np.zeros(X.shape[0])
        for label in self.regressors_:
            mask = (y_class_pred == label)
            predictions[mask] = self.regressors_[label].predict(X[mask])
        return predictions

if __name__ == "__main__":
    
    # Example dataset
    np.random.seed(42)
    X = np.random.rand(100, 5)  # 100 samples, 5 features
    y = np.random.randint(18, 80, size=100)  # Random ages between 18 and 80

    model = AgeGroupedRegressor(age_threshold=40)

    model.fit(X, y)

    y_pred = model.predict(X)

    print("Predictions:", y_pred)
