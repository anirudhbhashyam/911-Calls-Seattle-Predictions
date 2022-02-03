import os

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import utility as ut

DATA_PATH = os.path.abspath("../data/train-test")
DATA_TRAIN = "final_data_mean"
DATA_TEST = "final_data_mean_test"
DATA_EXT = "csv"
LABEL = "calls"

data = pd.read_csv(os.path.join(DATA_PATH, ".".join([DATA_TRAIN, DATA_EXT])), header = 0)
test_data = pd.read_csv(os.path.join(DATA_PATH, ".".join([DATA_TEST, DATA_EXT])), header = 0)

Y = data.pop(LABEL)
Y_test = test_data.pop(LABEL)

X_train, X_val, Y_train, Y_val = train_test_split(data, Y, shuffle = True, random_state = 43)

reg = GradientBoostingRegressor(n_iter_no_change = 5,
                                validation_fraction = 0.2)
reg.fit(X_train, Y_train)

predictions = reg.predict(test_data)

print(pd.DataFrame(np.vstack([np.floor(predictions), Y_test.to_numpy()]).T, columns = ["Predictions", "Real"]))
print(f"{ut.lossy_error(np.floor(predictions), Y_test) * 100:.2f}")





