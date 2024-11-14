
################################### KNN Model ########################################################################

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Task 2

data = [
    ['Black', 1, 1],
    ['Blue', 0, 0],
    ['Blue', -1, -1]
]
df = pd.DataFrame(data, columns=['y', 'x1', 'x2'])

X = df[['x1', 'x2']].values
y = df['y'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_X_train= scaler.fit_transform(X_train)
scaled_X_test= scaler.transform(X_test)



# Task 3
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)



# Task 4
new_observation = np.array([[0.1, 0.1]])
scaled_new_observation= scaler.transform(new_observation)
predicted_label  = knn.predict(scaled_new_observation)

print(f"Pedicted label for (x1=0.1, x2=0.1): {predicted_label[0]}")



# Task 5
predicted_proba = knn.predict_proba(new_observation)

prob_black = predicted_proba[0][list(knn.classes_).index('Black')]
prob_blue = predicted_proba[0][list(knn.classes_).index('Blue')]

print(f"Probability that the target variable is 'Black': {prob_black:.2f}")
print(f"Probability that the target variable is 'Blue': {prob_blue:.2f}")



################################# Tree-Based ##############################################################


df = pd.read_csv("C:/Users/Palvi/Desktop/Fall Semester/INSY 662 - Prof Han/Individual Assignment 2/Sheet4.csv")
df = df.dropna()
df = df.drop(columns=['Name'])
df = pd.get_dummies(df, columns=['Manuf', 'Type'], drop_first=True)


X = df.drop(columns=['Rating_Binary'])
y = df['Rating_Binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



##
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_features': [3, 4, 5, 6],
    'min_samples_leaf': [1, 2, 3, 4]
}


rf = RandomForestClassifier(random_state=0)
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid)
grid_search_rf.fit(X_train_scaled, y_train)


best_params_rf = grid_search_rf.best_params_
best_score_rf = grid_search_rf.best_score_
print(f"Best Random Forest Model Performance (accuracy score): {best_score_rf}")
print(f"Best combination of hyperparameters: {best_params_rf}")


##
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(random_state=0)
grid_search_gb = GridSearchCV(estimator=gb, param_grid=param_grid)
grid_search_gb.fit(X_train_scaled, y_train)


best_params_gb = grid_search_gb.best_params_
best_score_gb = grid_search_gb.best_score_
print(f"Best Gradient Boosting Model Performance (accuracy score): {best_score_gb}")
print(f"Best combination of hyperparameters: {best_params_gb}")




##
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


best_rf = RandomForestClassifier(**best_params_rf, random_state=0)
best_rf.fit(X_train_scaled, y_train)

best_gb = GradientBoostingClassifier(**best_params_gb, random_state=0)
best_gb.fit(X_train_scaled, y_train)


rf_test_accuracy = best_rf.score(X_test_scaled, y_test)
gb_test_accuracy = best_gb.score(X_test_scaled, y_test)
print(f"Random Forest Test Accuracy: {rf_test_accuracy}")
print(f"Gradient Boosting Test Accuracy: {gb_test_accuracy}")




##
import matplotlib.pyplot as plt

if rf_test_accuracy > gb_test_accuracy:
    feature_importances = best_rf.feature_importances_
else:
    feature_importances = best_gb.feature_importances_

# Ploting the feature importance
features = X.columns
indices = np.argsort(feature_importances)
plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), feature_importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.title('Feature Importance')
plt.show()
