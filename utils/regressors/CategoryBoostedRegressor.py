### Gender-Ethnicity boosted Regressor
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin

class CategoryBoostedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, category_column, base_regressor=LinearRegression):
        self.category_column = category_column
        self.base_regressor = base_regressor
        self.category_regressors = {}

    def fit(self, X, y):
        unique_categories = X[self.category_column].unique()

        # Train a separate regressor for each unique category
        for category in unique_categories:
            
            category_mask = X[self.category_column] == category
            
            category_X = X[category_mask].drop(columns=[self.category_column])
            category_y = y[category_mask]
            
            category_regressor = self.base_regressor().fit(category_X, category_y)
            self.category_regressors[category] = category_regressor

        return self

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])  # Initialize predictions array

        # Predict for each category
        for category, regressor in self.category_regressors.items():
            
            category_mask = X[self.category_column] == category
            category_X = X[category_mask].drop(columns=[self.category_column])

            if not category_X.empty:
                category_y_pred = regressor.predict(category_X)
                y_pred[category_mask] = category_y_pred

        return y_pred
    
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def get_params(self, deep=True):
        return {'category_column': self.category_column, 'base_regressor': self.base_regressor}

    def get_regressor(self, category):    
        return self.category_regressors.get(category, None)