
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

#model evaluation function
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

def model_evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return r2, mae




df = pd.read_csv('reqs_day_data.csv')

# Replacing Nulls with 0
df['count_of_driver_activations_per_day'] = df['count_of_driver_activations_per_day'].fillna(0)
date = df['CreatedDate']
df = df.drop(['CreatedDate'], axis=1)

# Train Test Split
X = df.drop(['reqs_per_day'], axis=1)
y = df['reqs_per_day']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


rf_reg = RandomForestRegressor(n_estimators=1000).fit(X_train, y_train)
cv_r2 = cross_val_score(rf_reg, X_train, y_train, cv = 10)
y_preds = rf_reg.predict(X_test)
cv_r2 = np.mean(cv_r2)
print('Cross Validation Score: ', cv_r2)
r2, mae = model_evaluate(y_test, y_preds)
print('r2 score: ', r2)
print('Mean Absolute Error: ', mae)

# Saving model to disk
pickle.dump(rf_reg, open('model.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))

print('Model Loaded to Disk')








