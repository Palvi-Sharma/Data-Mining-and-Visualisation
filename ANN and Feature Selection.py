
############################# Data Preprocessing ####################################################
import pandas as pd
df = pd.read_csv("OnlineNews.csv")
df.head(5)

df.info()
df.isnull().sum()

df = df.drop(['url', 'timedelta'], axis=1)
df.head()

df = pd.get_dummies(df, columns=['weekdays', 'data_channel'], drop_first = True, dtype= int)
print(df.dtypes)

X = df.drop(columns=["popularity"])
Y = df['popularity']

#Splitting Data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.3, random_state= 0 )


#Visualizing the data
import matplotlib.pyplot as plt

for column in df.columns:
    plt.hist(df[column], bins=30, alpha=0.7, color='blue')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()
    
for column in df.columns:
    plt.boxplot(df[column])
    plt.title(f'Box Plot of {column}')
    plt.ylabel(column)
    plt.show()


#Based on VISUALIZED analysis above, I found that the data is not normally distributed and have potential outliers
#Thus, decided to use MinMaxSaler 
#Standardize Data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)




################################ Optimal Nodes #########################################

#Finding the optimal nodes
from sklearn.neural_network import MLPClassifier

mlp= MLPClassifier(hidden_layer_sizes=(10),max_iter=1000, random_state=0)
model = mlp.fit(X_train_scaled,Y_train)
Y_test_pred= model.predict(X_test_scaled)

from sklearn.metrics import accuracy_score
accuracy_score(Y_test, Y_test_pred)
# output 0.6488145283336136

for i in [5, 10, 15, 20]:
    mlp = MLPClassifier(hidden_layer_sizes = (i), max_iter = 1000, random_state = 0)
    model = mlp.fit(X_train_scaled, Y_train) 
    Y_test_pred = model.predict(X_test_scaled)
    print(i,':',accuracy_score(Y_test, Y_test_pred))

#Output 
# 5  : 0.6408273078863292
# 10 : 0.6488145283336136
# 15 : 0.6504119724230705
# 20 : 0.6507482764419035

# All the accuracy scores are relatively close to each other
# 20 Nodes achieved the highest accuracy at 0.6507.





############################ Feature Selection: LASSO ###################################################################################################
from sklearn.linear_model import Lasso
ls = Lasso(alpha=0.01)
model = ls.fit(X_train_scaled,Y_train)

pd.DataFrame(list(zip(X_train_scaled.columns, model.coef_)), columns = ['predictor','coefficient'])

# Out[25]: 
#                      predictor  coefficient
# 0               n_tokens_title    -0.000000
# 1             n_tokens_content     0.000000
# 2              n_unique_tokens    -0.000000
# 3     n_non_stop_unique_tokens    -0.000000
# 4                    num_hrefs     0.000000
# 5               num_self_hrefs     0.000000
# 6                     num_imgs     0.000000
# 7                   num_videos     0.000000
# 8         average_token_length    -0.000000
# 9                 num_keywords     0.000000
# 10                  kw_min_min     0.000000
# 11                  kw_max_min     0.000000
# 12                  kw_avg_min     0.000000
# 13                  kw_min_max    -0.000000
# 14                  kw_max_max    -0.000000
# 15                  kw_avg_max     0.000000
# 16                  kw_min_avg     0.012581 # Selected
# 17                  kw_max_avg     0.000000
# 18                  kw_avg_avg     0.000000
# 19   self_reference_min_shares     0.000000
# 20   self_reference_max_shares     0.000000
# 21  self_reference_avg_sharess     0.000000
# 22                  is_weekend     0.115299 # Selected
# 23                      LDA_00     0.000000
# 24                      LDA_01    -0.000000
# 25                      LDA_02    -0.031286 # Selected
# 26                      LDA_03     0.000000
# 27                      LDA_04     0.000000
# 28         global_subjectivity     0.000000
# 29   global_sentiment_polarity     0.000000
# 30  global_rate_positive_words     0.000000
# 31  global_rate_negative_words    -0.000000
# 32         rate_positive_words     0.000000
# 33         rate_negative_words    -0.000000
# 34       avg_positive_polarity     0.000000
# 35       min_positive_polarity    -0.000000
# 36       max_positive_polarity     0.000000
# 37       avg_negative_polarity    -0.000000
# 38       min_negative_polarity    -0.000000
# 39       max_negative_polarity    -0.000000
# 40          title_subjectivity     0.000000
# 41    title_sentiment_polarity     0.000000
# 42             weekdays_Monday    -0.000000
# 43           weekdays_Saturday     0.000000
# 44             weekdays_Sunday     0.000000
# 45           weekdays_Thursday    -0.000000
# 46             weekdays_Tueday    -0.000000
# 47          weekdays_Wednesday    -0.000000
# 48  data_channel_Entertainment    -0.121275 # Selected
# 49      data_channel_Lifestyle     0.000000
# 50         data_channel_Others     0.000000
# 51   data_channel_Social Media     0.000000
# 52           data_channel_Tech     0.000000
# 53          data_channel_World    -0.130210 # Selected



######################### Feature Selection: Random Forest ##############################################

from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(random_state = 0)
model = randomforest.fit(X_train_scaled, Y_train)

feature_importance_df = pd.DataFrame(list(zip(X_train_scaled.columns, model.feature_importances_)), columns=['predictor', 'feature importance'])
feature_importance_df = feature_importance_df.sort_values(by ='feature importance', ascending = False)

feature_importance_df

# Out[31]: SELECT TOP 5
#                      predictor  feature importance
# 18                  kw_avg_avg            0.046188 # Selected
# 17                  kw_max_avg            0.041475 # Selected
# 25                      LDA_02            0.032978 # Selected
# 19   self_reference_min_shares            0.032057 # Selected
# 24                      LDA_01            0.030489 # Selected
# 15                  kw_avg_max            0.030280
# 27                      LDA_04            0.029646
# 12                  kw_avg_min            0.029297
# 23                      LDA_00            0.029255
# 3     n_non_stop_unique_tokens            0.028508
# 2              n_unique_tokens            0.028407
# 28         global_subjectivity            0.028299
# 8         average_token_length            0.027720
# 21  self_reference_avg_sharess            0.027625
# 1             n_tokens_content            0.027411
# 11                  kw_max_min            0.027213
# 26                      LDA_03            0.026938
# 30  global_rate_positive_words            0.026714
# 34       avg_positive_polarity            0.026161
# 29   global_sentiment_polarity            0.025429
# 20   self_reference_max_shares            0.024695
# 37       avg_negative_polarity            0.024040
# 31  global_rate_negative_words            0.023919
# 16                  kw_min_avg            0.023336
# 4                    num_hrefs            0.022071
# 33         rate_negative_words            0.021414
# 32         rate_positive_words            0.021043
# 13                  kw_min_max            0.017315
# 0               n_tokens_title            0.016678
# 38       min_negative_polarity            0.015124
# 6                     num_imgs            0.014982
# 39       max_negative_polarity            0.014571
# 41    title_sentiment_polarity            0.014105
# 5               num_self_hrefs            0.013551
# 35       min_positive_polarity            0.013531
# 40          title_subjectivity            0.013435
# 22                  is_weekend            0.011283
# 36       max_positive_polarity            0.010735
# 9                 num_keywords            0.010580
# 48  data_channel_Entertainment            0.009484
# 7                   num_videos            0.008368
# 14                  kw_max_max            0.007334
# 10                  kw_min_min            0.006196
# 52           data_channel_Tech            0.005983
# 53          data_channel_World            0.005967
# 51   data_channel_Social Media            0.005513
# 43           weekdays_Saturday            0.003329
# 45           weekdays_Thursday            0.003297
# 47          weekdays_Wednesday            0.003246
# 42             weekdays_Monday            0.003203
# 46             weekdays_Tueday            0.003189
# 44             weekdays_Sunday            0.002416
# 50         data_channel_Others            0.002237
# 49      data_channel_Lifestyle            0.001738




################################### Building ANN #################################################

# Model 1: Use all Predictors
import time
from sklearn.metrics import precision_score, recall_score

start_time = time.time()
mlp_1 =  MLPClassifier(hidden_layer_sizes = (20), max_iter = 1000, random_state = 0)
model_1 = mlp_1.fit(X_train_scaled, Y_train)
Y_test_pred_1 = model_1.predict(X_test_scaled)
end_time = time.time()
time_1 = end_time - start_time

accuracy_1 = accuracy_score(Y_test, Y_test_pred_1)
precision_1 = precision_score(Y_test, Y_test_pred_1)
recall_1 = recall_score(Y_test, Y_test_pred_1)

print("Model 1 (All Predictors):")
print(f"Accuracy: {accuracy_1}")
print(f"Precision: {precision_1}")
print(f"Recall: {recall_1}")
print(f"Training Time: {time_1} seconds\n")

#Output
# Model 1 (All Predictors):
# Accuracy: 0.6507482764419035
# Precision: 0.665083135391924
# Recall: 0.5771212641703882
# Training Time: 25.784961223602295 seconds




# Model 2: Use LASSO Feature Selection
lasso_features = ['kw_min_avg', 'is_weekend', 'LDA_02', 'data_channel_Entertainment', 'data_channel_World']
X_train_lasso = X_train_scaled[lasso_features]
X_test_lasso = X_test_scaled[lasso_features]

start_time2 = time.time()
mlp_2 = MLPClassifier(hidden_layer_sizes = (20), max_iter = 1000, random_state = 0)
model_2 = mlp_2.fit(X_train_lasso, Y_train)
Y_test_pred_2 = model_2.predict(X_test_lasso)
end_time2 = time.time()
time_2 = end_time2 - start_time2

accuracy_2 = accuracy_score(Y_test, Y_test_pred_2)
precision_2 = precision_score(Y_test, Y_test_pred_2)
recall_2 = recall_score(Y_test, Y_test_pred_2)

print("Model 2 (LASSO Selected Features):")
print(f"Accuracy: {accuracy_2}")
print(f"Precision: {precision_2}")
print(f"Recall: {recall_2}")
print(f"Training Time: {time_2} seconds\n")

#Output
# Model 2 (LASSO Selected Features):
# Accuracy: 0.6180427106103918
# Precision: 0.5959202039898005
# Recall: 0.6824115424252835
# Training Time: 1.3329424858093262 seconds




#Model 3: Use Random Forest Selection
rf_features = ['kw_avg_avg', 'kw_max_avg', 'LDA_02', 'self_reference_min_shares', 'LDA_01']
X_train_rf = X_train_scaled[rf_features]
X_test_rf = X_test_scaled[rf_features]

start_time3 = time.time()
mlp_3 = MLPClassifier(hidden_layer_sizes= (20), max_iter= 1000, random_state = 0)
model_3 = mlp_3.fit(X_train_rf, Y_train)
Y_test_pred_3 = model_3.predict(X_test_rf)
end_time3 = time.time()
time_3 = end_time3 - start_time3

accuracy_3 = accuracy_score(Y_test, Y_test_pred_3)
precision_3 = precision_score(Y_test, Y_test_pred_3)
recall_3 = recall_score(Y_test, Y_test_pred_3)

print("Model 3 (Random Forest-Selected Features):")
print(f"Accuracy: {accuracy_3}")
print(f"Precision: {precision_3}")
print(f"Recall: {recall_3}")
print(f"Training Time: {time_3} seconds")

#Output
# Model 3 (Random Forest-Selected Features):
# Accuracy: 0.6159408104926853
# Precision: 0.6082527624309392
# Recall: 0.6051185159738921
# Training Time: 3.407586097717285 seconds






