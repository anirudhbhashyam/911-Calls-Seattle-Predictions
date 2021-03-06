"""
Test
====
"""

import os
import pickle

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

import utility as ut
from variables import *


test_data = pd.read_csv(os.path.join(DATA_PATH, ".".join([DATA_TEST, DATA_EXT])), header = 0)
Y_test = test_data.pop(LABEL)

# CLASSES = np.unique(Y_test)
# N_CLASSES = len(CLASSES)
# Y_test = Y_test.replace(dict(zip(CLASSES, range(0, len(CLASSES)))))

# Data shape parameters.
N_FEATURES = test_data.shape[1]
N_SAMPLES = test_data.shape[0]
 
 
def test_nn(test_data: np.ndarray, Y_test: np.array) -> None:
	"""
	Provides testing data for the trained neural network.
	
 	Parameters
	----------
	test_data :
		The test data.
	Y_test :
		The test labels.

	Returns
	-------
	`None`
	"""
	print("Testing nn model against test data set.")
 
	try:
		model = tf.keras.models.load_model(os.path.join(NN_MODEL_SAVE_PATH, NN_MODEL_SAVE_NAME))
	except FileNotFoundError:
		print(f"Loaded model not found in {NN_MODEL_SAVE_PATH}. Check name or save path in variables.py.")
 
	predictions = model.predict(
    	np.array(test_data).reshape(-1, WB_SIZE, N_FEATURES)
	).flatten()
	# predictions = np.argmax(predictions, axis = 1)
 
	test_compare = pd.DataFrame(
    	np.vstack([predictions, Y_test.to_numpy()]).T, 
        columns = ["Predictions", "Real"])
 
	# Backtest plot.
	plt.figure(figsize = (16, 8))
	sns.lineplot(data = test_compare, x = list(range(test_compare.shape[0])), y = "Predictions")
	sns.lineplot(data = test_compare, x = list(range(test_compare.shape[0])), y = "Real")
	plt.title("NN Backtest Plot")
	plt.xlabel("Time Steps")
	plt.ylabel("Predictions")
	plt.legend(["Predicted Call Volume", "Real Call Volume"])
	plt.savefig(os.path.join(TEST_FIG_SAVE_PATH, ".".join(["nn_backtest_fig", FIG_EXT])), dpi = 160)
	
	print(f"Mean absolute test error (nn): {mean_absolute_percentage_error(Y_test, predictions):.2f}")
	print(f"Accuracy (nn): {(1 - ut.count_lossy_error(np.floor(predictions), Y_test, 5)) * 100:.2f} \u00b1 {5.0}\n")
 
 
def test_gb(test_data: np.ndarray, Y_test: np.array) -> None:
	"""
	Provides testing data for the sk learn model.
	
 	Parameters
	----------
	test_data :
		The test data.
	Y_test :
		The test labels.

	Returns
	-------
	`None`
	"""
	print("Testing sklearn model against test dataset.")
 
	try:
		with open(os.path.join(SK_MODEL_SAVE_PATH, ".".join([SK_MODEL_SAVE_NAME, SK_MODEL_SAVE_EXT])), "rb") as f:
			model = pickle.load(f)
	except FileNotFoundError:
		print(f"Loaded model not found in {SK_MODEL_SAVE_PATH}. Check name or save path in variables.py.")

  
	predictions = model.predict(
		test_data
	)
 
	test_compare = pd.DataFrame(
    	np.vstack([predictions, Y_test.to_numpy()]).T, 
        columns = ["Predictions", "Real"])

	# Backtest plot.
	plt.figure(figsize = (16, 8))
	sns.lineplot(data = test_compare, x = list(range(N_SAMPLES)), y = "Predictions")
	sns.lineplot(data = test_compare, x = list(range(N_SAMPLES)), y = "Real")
	plt.title("sklearn Backtest Plot")
	plt.xlabel("Time")
	plt.ylabel("Predictions")
	plt.legend(["Predicted Call Volume", "Real Call Volume"])
	plt.savefig(os.path.join(TEST_FIG_SAVE_PATH, ".".join(["gb_backtest_fig", FIG_EXT])), dpi = 160)
	
	print(f"Mean absolute test error (sk): {mean_absolute_percentage_error(Y_test, predictions):.2f}")
	print(f"Accuracy (sk): {(1 - ut.count_lossy_error(np.floor(predictions), Y_test, 5)) * 100:.2f} \u00b1 {5.0}\n")
	
    
 
 
def main():
	test_gb(test_data, Y_test)
	test_nn(test_data, Y_test)
	
	
if __name__ == "__main__":
	main()
 
	
	
	