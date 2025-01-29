import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

from utils.regressors import *

# Define the function to perform grid search and save results
def compare_models(X, y, results_file="fine_tuning_results.txt", random_state=None):
    
    # Define parameter grids
    param_grids = {
        'OrdinaryLinearRegressor': {
            'model': LinearRegression(),
            'param_grid': {}
        },
        'Lasso': {
            'model': Lasso(random_state=random_state),
            'param_grid': {'alpha': [0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [10000]}
        },
        'Ridge': {
            'model': Ridge(random_state=random_state),
            'param_grid': {'alpha': [0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [10000]}
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=random_state),
            'param_grid': {'n_estimators': [50, 100, 150], 'max_depth': [3, 5, 10], 'criterion': ['poisson']}
        },
        'MLPRegressor': {
            'model': MLPRegressor(max_iter=1000, random_state=random_state),
            'param_grid': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
    }

    scorer = 'neg_root_mean_squared_error'  # Directly use the built-in RMSE scorer

    # Save results to a file
    with open(results_file, "w") as file:
        for model_name, config in param_grids.items():
            model = config['model']
            param_grid = config['param_grid']
            
            print(f"Running GridSearchCV for {model_name}...")
            file.write(f"GridSearchCV Results for {model_name}:\n")
            
            try:
                grid_search = GridSearchCV(
                    model,
                    param_grid,
                    cv=3,
                    scoring=scorer,
                    refit=True,
                    n_jobs=-1,
                    verbose=2
                )
                grid_search.fit(X, y)
                
                # Save best parameters and score
                best_params = grid_search.best_params_
                best_score = -grid_search.best_score_  # neg_root_mean_squared_error returns negative values
                
                file.write(f"Best Parameters: {best_params}\n")
                file.write(f"Best Score (RMSE): {best_score:.4f}\n")
                
                # Save the entire GridSearchCV results as a DataFrame
                results_df = pd.DataFrame(grid_search.cv_results_)
                file.write("Top 5 Configurations:\n")
                top_5 = results_df.nsmallest(5, 'mean_test_score')
                file.write(top_5[['params', 'mean_test_score']].to_string(index=False))
                file.write("\n\n")
                
            except Exception as e:
                file.write(f"Error for {model_name}: {e}\n\n")

    print(f"Results saved to {results_file}")
