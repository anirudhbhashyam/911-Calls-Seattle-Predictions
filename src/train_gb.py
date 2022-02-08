import os

import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, RandomizedSearchCV

import utility as ut

DATA_PATH = os.path.relpath("../data/train-test")
DATA_TRAIN = "train_data"
DATA_TEST = "test_data"
DATA_EXT = "csv"
LABEL = "calls"

train_data = pd.read_csv(os.path.join(DATA_PATH, ".".join([DATA_TRAIN, DATA_EXT])), header = 0)
test_data = pd.read_csv(os.path.join(DATA_PATH, ".".join([DATA_TEST, DATA_EXT])), header = 0)

Y = train_data.pop(LABEL)
Y_test = test_data.pop(LABEL)

X_train, X_val, Y_train, Y_val = train_test_split(train_data, Y, shuffle = True, random_state = 43)

# Parameter space.
p_grid = {"alpha": np.arange(0.1, 1, 0.1), 
          "learning_rate": np.arange(0.01, 0.1, 0.02), 
          "ccp_alpha": np.arange(0, 2, 0.4), 
          "min_samples_leaf": range(1, 30), 
          "n_estimators": range(100, 500, 100)}


# cv = KFold(n_splits = 4, shuffle = True, random_state = 42) 
# outer_cv = KFold(n_splits = 4, shuffle = True, random_state = 43)

# Unnested parameter search and scoring.
# reg = GridSearchCV(estimator = GradientBoostingRegressor(), 
#                          param_grid = p_grid, 
#                          cv = cv, 
#                          verbose = 2,
#                          n_jobs = 4)
reg = GradientBoostingClassifier(n_estimators = 200, 
                                 random_state = 1299709,
                                 verbose = 1)
reg.fit(X_train, Y_train)
# scores = cross_val_score(reg, train_data, Y, cv = cv)

with open(os.path.join(SK_MODEL_SAVE_PATH, ".".join([SK_MODEL_SAVE_NAME, SK_MODEL_SAVE_EXT]))):
    
    pickle.dump(reg)

# predictions = reg.predict(test_data)

# print(pd.DataFrame(np.vstack([np.floor(predictions), Y_test.to_numpy()]).T, columns = ["Predictions", "Real"]))
# print(f"{ut.lossy_error(np.floor(predictions), Y_test) * 100:.2f}")





