import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

#load data
data = pd.read_csv('merged_data.csv', encoding='utf-8')

# 필요한 열 선택
selected_features = ['stn_id', 'borrowed_hour', 'borrowed_day', 'is_holiday', 'borrowed_num_nearby', '강수량(mm)', 'wind_chill', 'stn_gu', 'nearby_id', 'borrowed_date', 'borrowed_num']
data = data[selected_features]

# 범주형 데이터를 숫자로 변환 (Label Encoding)
categorical_features = ['stn_id', 'stn_gu', 'nearby_id']
for feature in categorical_features:
    data[feature] = pd.factorize(data[feature])[0]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(data.drop('borrowed_num', axis=1), data['borrowed_num'], test_size=0.2, random_state=42)

# LightGBM 모델 생성 및 훈련
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[test_data, train_data], callbacks=[
        lgb.early_stopping(stopping_rounds=3, verbose=100),
    ])


# 모델 평가
y_pred_train = model.predict(X_train, num_iteration=model.best_iteration)
y_pred_test = model.predict(X_test, num_iteration=model.best_iteration)

# 훈련 데이터에서의 평가
mse_train = mean_squared_error(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)


print(f'Training MSE: {mse_train}')
print(f'Training MAE: {mae_train}')
print(f'Training RMSE: {rmse_train}')


# 검증 데이터에서의 평가
mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)

print(f'Test MSE: {mse_test}')
print(f'Test MAE: {mae_test}')
print(f'Test RMSE: {rmse_test}')

# R-squared 계산
# 훈련 데이터에서의 R-squared 계산
r2_train = r2_score(y_train, y_pred_train)

# 테스트 데이터에서의 R-squared 계산
r2_test = r2_score(y_test, y_pred_test)

print("R-squared for training data:", r2_train)
print("R-squared for test data:", r2_test)