# Seoul Public Bike(따릉이) Demand Prediction

- [Seoul Public Bike(따릉이) Demand Prediction](#seoul-public-bike------demand-prediction)
  - [Project Overview](#project-overview)
  - [Objectives](#objectives)
  - [Execution Environment](#execution-environment)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Execution](#model-execution)
    - [Model Details](#model-details)
    - [Model Versions](#model-versions)
    - [Create and Activate Conda Environment](#create-and-activate-conda-environment)
    - [Install Required Packages](#install-required-packages)
    - [Launch Jupyter Notebook](#launch-jupyter-notebook)
  - [XGBoost and LightGBM Modeling](#xgboost-and-lightgbm-modeling)
    - [Data Preprocessing](#data-preprocessing-1)
    - [XGBoost Model](#xgboost-model)
    - [LightGBM Model](#lightgbm-model)
    - [Evaluation](#evaluation)
    - [Ensemble Modeling](#ensemble-modeling)
  - [Limitations & Conclusion](#limitations---conclusion)
    - [Limitations](#limitations)
    - [Conclusion](#conclusion)

## Project Overview

The Seoul Public Bike project has faced financial challenges, with deficits increasing over the years. The goal of this project is to leverage business analytics to understand the usage patterns, optimize system efficiency, and propose strategies for deficit reduction.

**For a detailed overview of the entire project, visit [this link](https://github.com/cyl0424/2023_BA_Project).**

## Objectives

1. **Financial Analysis:** Conduct a comprehensive analysis of the Seoul Public Bike project's financial status, identifying trends, and understanding the factors contributing to deficits.

2. **Usage Pattern Analysis:** Explore usage patterns of public bicycles, considering factors such as borrowed hour, borrowed day, and environmental conditions.

3. **Model Development:** Implement machine learning models, including LightGBM and XGBoost, to predict bicycle usage and optimize station-specific trends.

4. **Ensemble Model:** Combine the strengths of LightGBM and XGBoost through ensemble modeling to enhance prediction accuracy.

5. **Deficit Reduction Strategies:** Based on the analysis results, propose strategies to reduce the financial deficits associated with the public bicycle project.

## Execution Environment

- Python 3.11 or higher
- Conda (for managing the virtual environment)

## Data Preprocessing

- [Weather Preprocessing Details](https://github.com/jeonghyeonee/23_BA_preprocessing_modeling/Weather_preprocessing.md): Detailed explanation and code related to the preprocessing of weather data.

## Model Execution

### Model Details

- [XGBoost Model Details](https://github.com/jeonghyeonee/23_BA_preprocessing_modeling/XGBoost_modeling.md): Detailed explanation and code related to the XGBoost model.
- [LightGBM Model Details](https://github.com/jeonghyeonee/23_BA_preprocessing_modeling/LGBM_modeling.md): Detailed explanation and code related to the LightGBM model.
- [Ensemble Model Details](<(https://github.com/jeonghyeonee/23_BA_preprocessing_modeling/Ensemble_modeling.md)>): Detailed explanation and code related to the Ensemble model.

### Model Versions

- pandas==1.5.3
- numpy==1.24.3
- scikit-learn==1.3.0
- lightgbm==4.1.0
- xgboost==2.0.2

### Create and Activate Conda Environment

1. **Create Conda Environment:**

   ```bash
   conda create --name myenv python=3.11
   ```

   Replace `myenv` with the desired environment name.

2. **Activate Conda Environment:**
   ```bash
   conda activate myenv
   ```

### Install Required Packages

```bash
conda install --file requirements.txt
```

This command installs the necessary packages specified in the `requirements.txt` file within the Conda environment.

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Now, open the Jupyter Notebook and navigate to the `team7_ensemble_model.ipynb` notebook to run the code under the "1. Load and Preprocess Data" section.

Ensure that you are using Python 3.11 or a higher version and have activated your Conda environment before installing the required packages.

---

## XGBoost and LightGBM Modeling

In this section, we focus on the modeling aspect, utilizing both XGBoost and LightGBM to predict bicycle usage patterns in the Seoul Public Bike project.

### Data Preprocessing

Refer to the `weather_preprocessing.py` script for details on preprocessing weather data from the Korea Meteorological Administration. The processed data is stored in 'preprocessed_weather.csv'.

### XGBoost Model

XGBoost is employed to build a regression model for predicting bike rental counts. The model is configured with specific hyperparameters to optimize performance.

```python
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

### LightGBM Model

LightGBM, another powerful gradient boosting framework, is employed for regression. The model is trained with specific hyperparameters for optimal performance.

```python
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

### Evaluation

Both XGBoost and LightGBM models are evaluated based on relevant metrics such as RMSE (Root Mean Squared Error) and R-squared.

```python
# Evaluation - XGBoost Model
rmse_xgb_train = np.sqrt(mean_squared_error(y_train, y_pred_train_xgb))
rmse_xgb_test = np.sqrt(mean_squared_error(y_test, y_pred_test_xgb))

r2_xgb_train = r2_score(y_train, y_pred_train_xgb)
r2_xgb_test = r2_score(y_test, y_pred_test_xgb)

print("XGBoost Model Evaluation Metrics:")
print("Training RMSE:", rmse_xgb_train)
print("Testing RMSE:", rmse_xgb_test)
print("Training R-squared:", r2_xgb_train)
print("Testing R-squared:", r2_xgb_test)

# Evaluation - LightGBM Model
rmse_lgb_train = np.sqrt(mean_squared_error(y_train, y_pred_train_lgb))
rmse_lgb_test = np.sqrt(mean_squared_error(y_test, y_pred_test_lgb))

print("LightGBM Model Evaluation Metrics:")
print("Training RMSE:", rmse_lgb_train)
print("Testing RMSE:", rmse_lgb_test)
```

### Ensemble Modeling

Finally, we explore the ensemble modeling approach, combining the predictions from both XGBoost and LightGBM models.

```python
# Generate Predictions
y_pred_train_ensemble = (y_pred_train_xgb + y_pred_train_lgb) / 2
y_pred_test_ensemble = (y_pred_test_xgb + y_pred_test_lgb) / 2

# Evaluate Ensemble Model
ensemble_rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train_ensemble))
ensemble_rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test_ensemble))

ensemble_r2_train = r2_score(y_train, y_pred_train_ensemble)
ensemble_r2_test = r2_score(y_test, y_pred_test_ensemble)

print("Ensemble Model Evaluation Metrics:")
print("Training RMSE:", ensemble_rmse_train)
print("Testing RMSE:", ensemble_rmse_test)
print("Training R-squared:", ensemble_r2_train)
print("Testing R-squared:", ensemble_r2_test)
```

This ensemble approach provides a robust solution for predicting bike rental counts by leveraging the strengths of both XGBoost and LightGBM models.

Adjust hyperparameters and model settings based on the characteristics of your dataset for optimal performance.

---

## Limitations & Conclusion

### Limitations

1. **Device Limitations:**
   Due to the physical limitations of the training device, we encountered challenges in achieving the desired results. The device's computational power and memory capacity posed constraints, impacting the complexity of the models we could train.

2. **Data Size Restriction:**
   The sheer volume of available data posed a challenge. To mitigate this, we focused on training the models with a subset of the data. Specifically, due to resource constraints, we opted to use only winter data for training, potentially limiting the model's ability to generalize across seasons.

3. **Memory Errors and Resource Scarcity:**
   The training process was hampered by frequent memory errors and resource scarcity. The large dataset and complex models strained the available resources, leading to limitations in model training and hindering the exploration of the entire dataset.

### Conclusion

Despite the encountered limitations, our project provides valuable insights into Seoul Public Bike demand. The financial analysis, usage pattern exploration, and ensemble modeling with XGBoost and LightGBM contribute to a comprehensive understanding of the system. Moving forward, addressing the mentioned limitations by leveraging more powerful hardware or cloud computing resources could enhance the model's predictive capabilities across diverse datasets. Additionally, exploring alternative feature engineering techniques and refining model hyperparameters may further improve overall performance.
