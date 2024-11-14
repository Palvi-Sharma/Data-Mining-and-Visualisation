
###############################################################################################################

import pandas as pd

# Load the dataset
df = pd.read_csv('ToyotaCorolla.csv')
df = df.dropna()

# Task 0: Define predictors and target variable 
X = df.iloc[:,3:12]  
y = df['Price']  


# Task 1
descriptive_stats = X.describe()
print(descriptive_stats)

df = X.drop(columns=["Cylinders"])  # Droping Cylinders 


# Task 2
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=662)


# Task 3
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test= scaler.transform(X_test)

scaled_X_train = pd.DataFrame(scaled_X_train, columns=X_train.columns)
scaled_X_test = pd.DataFrame(scaled_X_test, columns=X_test.columns)


# Task 4:
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

from sklearn.metrics import mean_squared_error
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f"Linear Regression MSE: {mse_lr}")


# Task 5: Ridge regression
from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha=1)
ridge_model.fit(scaled_X_train, y_train)

y_pred_ridge = ridge_model.predict(scaled_X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print(f"Ridge Regression (alpha=1) MSE: {mse_ridge}")


# Task 6: LASSO regression 
from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha=1)
lasso_model.fit(scaled_X_train, y_train)

y_pred_lasso = lasso_model.predict(scaled_X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print(f"LASSO Regression (alpha=1) MSE: {mse_lasso}")

# Compare the MSEs
print(f"Linear Regression MSE: {mse_lr}")
print(f"Ridge Regression MSE: {mse_ridge}")
print(f"LASSO Regression MSE: {mse_lasso}")


# Task 7: Ridge and LASSO with different alpha values 
alphas = [10, 100, 1000, 10000]

for alpha in alphas:
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(scaled_X_train, y_train)
    mse_ridge = mean_squared_error(y_test, ridge_model.predict(scaled_X_test))
    print(f"Ridge Regression (alpha={alpha}) MSE: {mse_ridge}")
  
for alpha in alphas:    
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(scaled_X_train, y_train)
    mse_lasso = mean_squared_error(y_test, lasso_model.predict(scaled_X_test))
    print(f"LASSO Regression (alpha={alpha}) MSE: {mse_lasso}")


# Task 8: Coefficients at alpha = 10000
ridge_model_10000 = Ridge(alpha=10000)
ridge_model_10000.fit(X_train_scaled, y_train)
ridge_coeffs = ridge_model_10000.coef_
print(f"Ridge Coefficients (alpha=10000): {ridge_coeffs}")

lasso_model_10000 = Lasso(alpha=10000)
lasso_model_10000.fit(X_train_scaled, y_train)
lasso_coeffs = lasso_model_10000.coef_
print(f"LASSO Coefficients (alpha=10000): {lasso_coeffs}")
