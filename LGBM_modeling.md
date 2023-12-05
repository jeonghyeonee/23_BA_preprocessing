# LightGBM Modeling

In this section, we'll walk through the process of building a LightGBM regression model using the Korea Bike Sharing dataset. LightGBM is a gradient boosting framework designed for efficiency and speed.

## LightGBM Model

LightGBM is a widely-used machine learning algorithm, particularly for regression tasks. It constructs an ensemble of decision trees sequentially, with each tree aimed at correcting errors made by the preceding ones. Here, we employ the LightGBM library to construct a regression model for predicting bike rental counts.

### Parameters

Before training the LightGBM model, we specify the hyperparameters that control the learning process. These key parameters include:

- `objective`: Set to 'regression' for regression tasks.
- `metric`: The evaluation metric, 'rmse' (Root Mean Squared Error) in this case.
- `boosting_type`: Type of boosting model, set to 'gbdt' for tree-based models.
- `num_leaves`: Maximum number of leaves in a tree.
- `learning_rate`: The step size shrinkage used to prevent overfitting.
- `feature_fraction`: Fraction of features used for training trees.

### Training

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# LightGBM Parameters
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 80,
    'learning_rate': 0.03,
    'feature_fraction': 0.9
}

# Train-Test Data Split
train_data_lgb = lgb.Dataset(X_train, label=y_train)
test_data_lgb = lgb.Dataset(X_test, label=y_test, reference=train_data_lgb)

# Train LightGBM Model
lgb_model = lgb.train(lgb_params, train_data_lgb, num_boost_round=1000, valid_sets=[test_data_lgb], callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=50)])
```

### Prediction

After model training, we generate predictions on both the training and test datasets.

```python
# Generate Predictions
y_pred_train_lgb = lgb_model.predict(X_train, num_iteration=lgb_model.best_iteration)
y_pred_test_lgb = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
```

### Evaluation

Evaluate the performance of the LightGBM model using relevant metrics such as RMSE (Root Mean Squared Error).

```python
# Evaluate LightGBM Model
rmse_lgb_train = np.sqrt(mean_squared_error(y_train, y_pred_train_lgb))
rmse_lgb_test = np.sqrt(mean_squared_error(y_test, y_pred_test_lgb))

print("LightGBM Model Evaluation Metrics:")
print("Training RMSE:", rmse_lgb_train)
print("Testing RMSE:", rmse_lgb_test)
```

Adjust hyperparameters and model settings based on your dataset characteristics for optimal performance.
