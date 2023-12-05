# XGBoost Modeling

In this section, we'll cover the XGBoost modeling process using the Korea Bike Sharing dataset. XGBoost is an efficient and scalable implementation of gradient boosting framework.

## XGBoost Model

XGBoost is a popular machine learning algorithm that is well-suited for regression tasks. It works by training an ensemble of decision trees sequentially, where each tree corrects the errors made by the previous ones. Here, we use the XGBoost library to build a regression model for predicting bike rental counts.

### Parameters

Before training the XGBoost model, we define the hyperparameters that control the learning process. These parameters include:

- `objective`: Set to 'reg:squarederror' for regression tasks.
- `eval_metric`: The evaluation metric, in this case, 'rmse' (Root Mean Squared Error).
- `booster`: Type of boosting model, set to 'gbtree' for tree-based models.
- `learning_rate`: The step size shrinkage used to prevent overfitting.
- `max_depth`: Maximum depth of a tree.
- `subsample`: Fraction of samples used for training trees.
- `device`: Specify 'gpu' to use GPU acceleration.

### Training

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# XGBoost Parameters
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'booster': 'gbtree',
    'learning_rate': 0.1,
    'max_depth': 13,
    'subsample': 0.8,
    'device': 'gpu'
}

# Train-Test Data Split
train_data_xgb = xgb.DMatrix(X_train, label=y_train)
test_data_xgb = xgb.DMatrix(X_test, label=y_test)

# Train XGBoost Model
xgb_model = xgb.train(xgb_params, train_data_xgb, num_boost_round=10000, evals=[(test_data_xgb, 'eval')], early_stopping_rounds=3, verbose_eval=100)
```

### Prediction

After training the model, we can generate predictions on both the training and test datasets.

```python
# Generate Predictions
y_pred_train_xgb = xgb_model.predict(train_data_xgb)
y_pred_test_xgb = xgb_model.predict(test_data_xgb)
```

### Evaluation

Finally, we evaluate the performance of the XGBoost model using relevant metrics such as RMSE (Root Mean Squared Error) and R-squared.

```python
# Evaluate XGBoost Model
rmse_xgb_train = np.sqrt(mean_squared_error(y_train, y_pred_train_xgb))
rmse_xgb_test = np.sqrt(mean_squared_error(y_test, y_pred_test_xgb))

r2_xgb_train = r2_score(y_train, y_pred_train_xgb)
r2_xgb_test = r2_score(y_test, y_pred_test_xgb)

print("XGBoost Model Evaluation Metrics:")
print("Training RMSE:", rmse_xgb_train)
print("Testing RMSE:", rmse_xgb_test)
print("Training R-squared:", r2_xgb_train)
print("Testing R-squared:", r2_xgb_test)
```

Adjust the hyperparameters and model settings based on the characteristics of your dataset for optimal performance.
