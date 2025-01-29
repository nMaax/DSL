import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class Node:
    def __init__(self, feature_index=None, threshold=None, regressor=None, depth=0):
        self.feature_index = feature_index
        self.threshold = threshold
        self.regressor = regressor
        self.depth = depth
        self.left = None
        self.right = None

class TasselatedLinearRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, 
                 base_regressor=LinearRegression(), 
                 max_depth=5, 
                 min_reduction=0, 
                 random_state=None, 
                 threshold_strategy="linspace", 
                 num_thresholds=10):
        """
        Initialize the model.
        """
        self.base_regressor = base_regressor or LinearRegression()
        self.max_depth = max_depth
        self.min_reduction = min_reduction
        self.random_state = random_state
        self.threshold_strategy = threshold_strategy
        self.num_thresholds = num_thresholds
        self.root = None

    def fit(self, X, y):
        """Fits the tree to the training data."""
        X, y = self._convert_to_numpy(X, y)
        print('Fit started...')
        self.root = self._build_tree(X, y, depth=0)
        print('Fit finished')
        return self

    def predict(self, X):
        """Predicts values for input data."""
        X = self._convert_to_numpy(X)
        return np.array([self._predict_single(x, self.root) for x in X])

    def _predict_single(self, x, node):
        """Explore the tree for a single prediction."""
        if node.left is None and node.right is None:  # Leaf node
            return node.regressor.predict(x.reshape(1, -1))[0]
        if x[node.feature_index] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def _get_thresholds(self, feature_values):
        """
        Get thresholds based on the chosen strategy.
        """
        if self.threshold_strategy == "linspace":
            # Evenly spaced thresholds
            return np.linspace(feature_values.min(), feature_values.max(), num=self.num_thresholds)
        
        elif self.threshold_strategy == "midpoints":
            # Midpoints between unique feature values
            unique_values = np.unique(feature_values)
            return (unique_values[:-1] + unique_values[1:]) / 2
        
        elif self.threshold_strategy == "quantiles":
            # Quantile-based thresholds
            quantiles = np.linspace(0, 1, num=self.num_thresholds)
            return np.quantile(feature_values, quantiles)
        
        else:
            raise ValueError(f"Unknown threshold_strategy: {self.threshold_strategy}")

    def _partition(self, feature_index, threshold, X, y):
        """Partitions data based on a feature index and threshold."""
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    def _build_tree(self, X, y, depth, split='8020'): # TODO split='loo' (leave-one-out)
        """Recursively builds the tree."""
        if len(y) < 2:  # Not enough data to split
            raise ValueError("Insufficient data to split during tree building.")

        if split == '8020':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
        elif split == 'lou':
            pass
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        if len(y_train) == 0 or len(y_test) == 0:  # Handle empty splits
            raise ValueError("Split resulted in empty datasets.")

        # Train a base regressor on the current node
        regressor = clone(self.base_regressor)
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
        base_error = mean_squared_error(y_test, y_pred)

        best_reduction = -np.inf
        best_feature_index = None
        best_threshold = None

        if depth > self.max_depth:
            print('Found Leaf Node')
            leaf = Node(regressor=regressor, depth=depth)
            return leaf

        for feature_index in range(X.shape[1]):
            feature_values = X_train[:, feature_index]
            thresholds = self._get_thresholds(feature_values)
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self._partition(feature_index, threshold, X_train, y_train)

                if len(y_left) < 2 or len(y_right) < 2:  # Avoid insufficient data for sub-splits
                    continue
                
                if split == '8020':
                    X_left_train, X_left_test, y_left_train, y_left_test = train_test_split(
                        X_left, y_left, test_size=0.2, random_state=self.random_state
                    )
                elif split == 'lou':
                    pass
                else:
                    X_left_train, X_left_test, y_left_train, y_left_test = X_left, X_left, y_left, y_left
                
                if split == '8020':
                    X_right_train, X_right_test, y_right_train, y_right_test = train_test_split(
                        X_right, y_right, test_size=0.2, random_state=self.random_state
                    )
                elif split == 'lou':
                    pass
                else:
                    X_right_train, X_right_test, y_right_train, y_right_test = X_right, X_right, y_right, y_right

                if len(y_left_train) == 0 or len(y_right_train) == 0:
                    continue

                regressor_left = clone(self.base_regressor)
                regressor_right = clone(self.base_regressor)

                regressor_left.fit(X_left_train, y_left_train)
                regressor_right.fit(X_right_train, y_right_train)

                y_pred_left = regressor_left.predict(X_left_test)
                y_pred_right = regressor_right.predict(X_right_test)

                error_left = mean_squared_error(y_left_test, y_pred_left)
                error_right = mean_squared_error(y_right_test, y_pred_right)

                reduction = base_error - (error_left + error_right)

                if reduction > best_reduction:
                    best_reduction = reduction
                    best_feature_index = feature_index
                    best_threshold = threshold

        if best_reduction > self.min_reduction:
            print (f'Found split at feature f{best_feature_index} on value {best_threshold}, which lead a reduction on MSE of {best_reduction}')
            X_left, y_left, X_right, y_right = self._partition(best_feature_index, best_threshold, X, y)

            if len(y_left) == 0 or len(y_right) == 0:  # Avoid splits that generate empty datasets
                raise ValueError("Partitioning resulted in empty datasets.")

            node = Node(feature_index=best_feature_index, threshold=best_threshold, regressor=regressor, depth=depth)
            
            node.left = self._build_tree(X_left, y_left, depth=depth+1, split=split)
            node.right = self._build_tree(X_right, y_right, depth=depth+1, split=split)
            return node
        else:
            # Create a leaf node
            print('Found Leaf Node')
            leaf = Node(regressor=regressor, depth=depth)
            return leaf

    def _convert_to_numpy(self, X, y=None):
        """Converts data to NumPy arrays for processing."""
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if y is not None and isinstance(y, pd.Series):
            y = y.to_numpy()
        return (X, y) if y is not None else X
    
    def get_tasselation(self, node=None):
        """
        Retrieves the tree (tessellation) structure as a dictionary.
        Each node contains its feature index, threshold, and regressor info.
        """
        if node is None:
            node = self.root

        if node is None:
            return None
        
        if node.left is None and node.right is None:  # Leaf node
            return {"type": "leaf", "regressor": str(node.regressor)}
        
        return {
            "type": "node",
            "feature_index": node.feature_index,
            "threshold": node.threshold,
            "left": self.get_tasselation(node.left),
            "right": self.get_tasselation(node.right),
        }

if __name__ == "__main__":
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=1000, n_features=10, noise=0.5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = TasselatedLinearRegressor(threshold_strategy='quantiles')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #print("Predictions:", predictions)

    print(mean_squared_error(y_pred, y_test))
    print(model.get_tasselation())

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    #print("Predictions:", predictions)

    print(mean_squared_error(y_pred, y_test))