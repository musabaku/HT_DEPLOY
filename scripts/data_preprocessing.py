import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def create_multiclass_target(y):
    return np.where((y[:, 0] == 0) & (y[:, 1] == 0), 0,
                    np.where((y[:, 0] == 1) & (y[:, 1] == 1), 1,
                             np.where((y[:, 0] == 0) & (y[:, 1] == 1), 2, 3)))

df = pd.read_csv(r'C:\Users\musab\Desktop\project\data\ht_traning_1_csv.csv')  
df.dropna(inplace=True)
y = df.iloc[:, :2].values
X = df.iloc[:, 2:].values

y_multiclass = create_multiclass_target(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_multiclass, test_size=0.2, random_state=42)

data_dir = os.path.join(os.getcwd(), 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

np.save(os.path.join(data_dir, 'rf_X_train.npy'), X_train)
np.save(os.path.join(data_dir, 'xgb_X_train.npy'), X_train)
np.save(os.path.join(data_dir, 'lgbm_X_train.npy'), X_train)

np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
np.save(os.path.join(data_dir, 'X_test.npy'), X_test)
np.save(os.path.join(data_dir, 'y_test.npy'), y_test)

