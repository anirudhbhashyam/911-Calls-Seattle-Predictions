import os

import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
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


NUM_TRIALS = 10
# Parameter space.
p_grid = {"alpha": np.arange(0.01, 1, 0.1), 
          "learning_rate": np.arange(0.01, 0.1, 0.02), 
          "ccp_alpha": np.arange(0, 2, 0.1), 
          "min_samples_leaf": range(1, 30), 
          "n_estimators": range(100, 500, 100)}

# Arrays to store scores.
non_nested_scores = np.zeros(NUM_TRIALS)
nested_scores = np.zeros(NUM_TRIALS)

inner_cv = KFold(n_splits = 4, shuffle = True, random_state = 42) 
# outer_cv = KFold(n_splits = 4, shuffle = True, random_state = 43)

# Unnested parameter search and scoring.
reg = RandomizedSearchCV(estimator = GradientBoostingRegressor(), 
                         param_distributions = p_grid, 
                         cv = inner_cv, 
                         n_iter = 100)
reg.fit(train_data, Y)
# non_nested_scores[i] = clf.best_score_

predictions = reg.predict(test_data)

print(pd.DataFrame(np.vstack([np.floor(predictions), Y_test.to_numpy()]).T, columns = ["Predictions", "Real"]))
print(f"{ut.lossy_error(np.floor(predictions), Y_test) * 100:.2f}")





