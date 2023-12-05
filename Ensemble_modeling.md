# Ensemble Modeling with LightGBM and XGBoost

In this section, we'll explore the ensemble modeling approach using both LightGBM and XGBoost for the Korea Bike Sharing dataset. Ensemble models combine predictions from multiple individual models to enhance overall performance.

## Data Preparation

Start by loading the dataset and selecting relevant features for modeling.

```python
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Load Data
data = pd.read_csv('merged_data.csv', encoding='utf-8')

# Selected Features
selected_features = ['stn_id', 'borrowed_hour', 'borrowed_day', 'is_holiday', 'borrowed_num_nearby', '강수량(mm)', 'wind_chill', 'nearby_id', 'borrowed_date', 'borrowed_num']
data = data[selected_features]

# Label Encoding
categorical_features = ['stn_id', 'nearby_id']
for feature in categorical_features:
    data[feature] = pd.factorize(data[feature])[0]

# Train-Test Data Split
X_train, X_test, y_train, y_test = train_test_split(data.drop('borrowed_num', axis=1), data['borrowed_num'], test_size=0.2, random_state=42)
```

## LightGBM Model

Next, we'll train a LightGBM model using both 'gbdt' and 'dart' boosting types.

```python
# LightGBM Model - gbdt
lgb_params_gbdt = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 80,
    'learning_rate': 0.05,
    'feature_fraction': 1.0,
    'device': 'gpu'
}

# LightGBM Model - dart
lgb_params_dart = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'dart',
    'num_leaves': 80,
    'learning_rate': 0.05,
    'feature_fraction': 1.0,
    'device': 'gpu'
}

train_data_lgb = lgb.Dataset(X_train, label=y_train)
test_data_lgb = lgb.Dataset(X_test, label=y_test, reference=train_data_lgb)

lgb_model_gbdt = lgb.train(lgb_params_gbdt, train_data_lgb, num_boost_round=10000, valid_sets=[test_data_lgb, train_data_lgb], callbacks=[lgb.early_stopping(stopping_rounds=3, verbose=100)])

lgb_model_dart = lgb.train(lgb_params_dart, train_data_lgb, num_boost_round=10000, valid_sets=[test_data_lgb, train_data_lgb], callbacks=[lgb.early_stopping(stopping_rounds=3, verbose=100)])
```

## XGBoost Model

Now, we'll train an XGBoost model.

```python
# XGBoost Model
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'booster': 'gbtree',
    'learning_rate': 0.1,
    'max_depth': 13,
    'subsample': 0.8,
    'device': 'gpu'
}

train_data_xgb = xgb.DMatrix(X_train, label=y_train)
test_data_xgb = xgb.DMatrix(X_test, label=y_test)

xgb_model = xgb.train(xgb_params, train_data_xgb, num_boost_round=10000, evals=[(test_data_xgb, 'eval')], early_stopping_rounds=3, verbose_eval=100)
```

## Ensemble Modeling

After training individual models, we'll create ensemble predictions using a weighted average approach.

```python
# Generate Predictions
y_pred_train_lgb_gbdt = lgb_model_gbdt.predict(X_train, num_iteration=lgb_model_gbdt.best_iteration)
y_pred_test_lgb_gbdt = lgb_model_gbdt.predict(X_test, num_iteration=lgb_model_gbdt.best_iteration)

y_pred_train_lgb_dart = lgb_model_dart.predict(X_train, num_iteration=lgb_model_dart.best_iteration)
y_pred_test_lgb_dart = lgb_model_dart.predict(X_test, num_iteration=lgb_model_dart.best_iteration)

y_pred_train_xgb = xgb_model.predict(train_data_xgb)
y_pred_test_xgb = xgb_model.predict(test_data_xgb)

# Calculate RMSE for each model
rmse_lgb_gbdt = np.sqrt(mean_squared_error(y_train, y_pred_train_lgb_gbdt))
rmse_lgb_dart = np.sqrt(mean_squared_error(y_train, y_pred_train_lgb_dart))
rmse_xgb = np.sqrt(mean_squared_error(y_train, y_pred_train_xgb))

# Set weights based on RMSE
weight_lgb_gbdt = 1 / (1 + rmse_lgb_gbdt)
weight_lgb_dart = 1 / (1 + rmse_lgb_dart)
weight_xgb = 1 / (1 + rmse_xgb)

# Weighted Ensemble Predictions
y_pred_train_ensemble_weighted = (weight_lgb_gbdt * y_pred_train_lgb_gbdt) + (weight_lgb_dart * y_pred_train_lgb_dart) + (weight_xgb * y_pred_train_xgb)
y_pred_test_ensemble_weighted = (weight_lgb_gbdt * y_pred_test_lgb_gbdt) + (weight_lgb_dart * y_pred_test_lgb_dart) + (weight_xgb * y_pred_test_xgb)

# Simple Ensemble Predictions (Equal weights)
y_pred_train_ensemble_simple = (y_pred_train_lgb_gbdt + y_pred_train_lgb_dart + y_pred_train_xgb) / 3
y_pred_test_ensemble_simple = (y_pred_test_lgb_gbdt + y_pred_test_lgb_dart + y_pred_test_xgb) / 3

# Evaluate Ensemble Models
ensemble_rmse_weighted_train = np.sqrt(mean_squared_error(y_train, y_pred_train_ensemble_weighted))
ensemble_rmse_weighted_test = np.sqrt(mean_squared_error(y_test, y_pred_test_ensemble_weighted))

ensemble_rmse_simple_train = np.sqrt(mean_squared_error(y_train, y_pred_train_ensemble_simple))
ensemble_rmse_simple_test = np.sqrt(mean_squared_error(y_test, y_pred_test_ensemble_simple))

ensemble_r2_weighted_train = r2_score(y_train, y_pred_train_ensemble_weighted)
ensemble_r2_weighted_test = r2_score(y_test, y_pred_test_ensemble_weighted)

ensemble_r2_simple_train = r2_score(y_train, y_pred_train_ensemble_simple)
ensemble_r2_simple_test = r2_score(y_test, y_pred_test_ensemble_simple)

print("Weighted Ensemble Model Evaluation Metrics:")
print("Training RMSE:", ensemble_rmse_weighted_train)
print("Testing RMSE:", ensemble_rmse_weighted_test)
print("Training R-squared:", ensemble_r2_weighted_train)
print("Testing R-squared:", ensemble_r2_weighted_test)

print("\nSimple Ensemble Model Evaluation Metrics (Equal Weights):")
print("Training RMSE:", ensemble_rmse_simple_train)
print("Testing RMSE:", ensemble_rmse_simple_test)
print("Training R-squared:", ensemble_r2_simple_train)
print("Testing R-squared:", ensemble_r2_simple_test)

# Individual Model Evaluation
lgb_rmse_gbdt_train = np.sqrt(mean_squared_error(y_train, y_pred_train_lgb_gbdt))
lgb_r2_gbdt_train = r2_score(y_train, y_pred_train_lgb_gbdt)
lgb_rmse_gbdt_test = np.sqrt(mean_squared_error(y_test, y_pred_test_lgb_gbdt))
lgb_r2_gbdt_test = r2_score(y_test, y_pred_test_lgb_gbdt)

lgb_rmse_dart_train = np.sqrt(mean_squared_error(y_train, y_pred_train_lgb_dart))
lgb_r2_dart_train = r2_score(y_train, y_pred_train_lgb_dart)
lgb_rmse_dart_test = np.sqrt(mean_squared_error(y_test, y_pred_test_lgb_dart))
lgb_r2_dart_test = r2_score(y_test, y_pred_test_lgb_dart)

xgb_rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train_xgb))
xgb_r2_train = r2_score(y_train, y_pred_train_xgb)
xgb_rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test_xgb))
xgb_r2_test = r2_score(y_test, y_pred_test_xgb)

print("\nIndividual Model Evaluation Metrics:")
print("LightGBM - gbdt Model:")
print("  Training RMSE:", lgb_rmse_gbdt_train)
print("  Testing RMSE:", lgb_rmse_gbdt_test)
print("  Training R-squared:", lgb_r2_gbdt_train)
print("  Testing R-squared:", lgb_r2_gbdt_test)

print("\nLightGBM - dart Model:")
print("  Training RMSE:", lgb_rmse_dart_train)
print("  Testing RMSE:", lgb_rmse_dart_test)
print("  Training R-squared:", lgb_r2_dart_train)
print("  Testing R-squared:", lgb_r2_dart_test)

print("\nXGBoost Model:")
print("  Training RMSE:", xgb_rmse_train)
print("  Testing RMSE:", xgb_rmse_test)
print("  Training R-squared:", xgb_r2_train)
print("  Testing R-squared:", xgb_r2_test)

# Prediction on New Data
new_data = pd.read_csv('data/new_data.csv', encoding='utf-8')
new_data = new_data[selected_features[:-1]]

for feature in categorical_features:
    new_data[feature] = pd.factorize(new_data[feature])[0]

y_pred_lgb_gbdt_new = lgb_model_gbdt.predict(new_data, num_iteration=lgb_model_gbdt.best_iteration, predict_disable_shape_check=True)
y_pred_lgb_dart_new = lgb_model_dart.predict(new_data, num_iteration=lgb_model_dart.best_iteration, predict_disable_shape_check=True)
y_pred_xgb_new = xgb_model.predict(xgb.DMatrix(new_data))

# Ensemble Prediction on New Data
y_pred_ensemble_new = (weight_lgb_gbdt * y_pred_lgb_gbdt_new) + (weight_lgb_dart * y_pred_lgb_dart_new) + (weight_xgb * y_pred_xgb_new)

# Display or Save Predictions
new_data['LGBM_gbdt_Prediction'] = y_pred_lgb_gbdt_new
new_data['LGBM_dart_Prediction'] = y_pred_lgb_dart_new
new_data['XGB_Prediction'] = y_pred_xgb_new
new_data['Ensemble_Prediction'] = y_pred_ensemble_new

print("\nPredictions on New Data:")
print(new_data[['LGBM_gbdt_Prediction', 'LGBM_dart_Prediction', 'XGB_Prediction', 'Ensemble_Prediction']])
# Or save to a CSV file
new_data.to_csv('predicted_results_ensemble.csv', index=False)
```

Adjust hyperparameters and model settings as needed for optimal performance on your specific dataset. This ensemble approach combines the strengths of LightGBM and XGBoost models, providing a robust solution for predicting bike rental counts.
