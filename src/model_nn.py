import os
from typing import Union

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error

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

sample_weights = np.ones(Y.shape[0])
sample_weights[np.logical_or(train_data["hour"] > 10, train_data["hour"] <= 22)] = 1.5

N_FEATURES = train_data.shape[1]
N_SAMPLES = train_data.shape[0]

def build_and_compile(input_: tuple = (1, N_FEATURES)):
	"""
	Build and compile a TensorFLow LSTM network.
	
	Parameters
	----------
	input_ :
		Shape of the trainining data. Should specify 
		`(batch_size, n_features)`
  
	Returns
	-------
	`tf.Model` :
		A compiled TensorFlow model.
	"""
 
	# Initialise input tensor.
	# input_shape = tf.keras.Input(
	# 	shape = input_
	# )

	# lstm = tf.keras.layers.LSTM(100,
    #                          activation = "relu")(input_shape)

	# # First hidden layer with 512 neurons.
	# hidden_1 = tf.keras.layers.Dense(
	# 	512,
	# 	activation = "relu",
	# 	name = "hidden_1"
	# )(lstm)

 
	# # Second hidden layer with 128 neurons.
	# hidden_2 = tf.keras.layers.Dense(
	# 	128,
	# 	activation = "linear",
	# 	name = "hidden_2"
	# )(hidden_1)

	# # Third hidden layer with 64 neurons.
	# hidden_3 = tf.keras.layers.Dense(
	# 	64,
	# 	activation = "linear",
	# 	name = "hidden_3"
	# )(hidden_2)

	# # Final output dense layer.
	# predictions = tf.keras.layers.Dense(
	# 	1,
	# 	activation = "linear",
	# 	name = "predictions"
	# )(hidden_3) 

	# # Combine layers.
	# model = tf.keras.models.Model(
	# 	inputs = input_shape,
	# 	outputs = predictions
	# )
 
	model = tf.keras.models.Sequential([
    	tf.keras.layers.LSTM(500, input_shape = input_, return_sequences = True, name = "lstm_1"),
    	tf.keras.layers.LSTM(500, return_sequences = True, name = "lstm_2"),
    	tf.keras.layers.LSTM(50, name = "lstm_3"),
		tf.keras.layers.Dense(512, name = "hidden_1"),
		tf.keras.layers.Dense(128, name = "hidden_2"),
		tf.keras.layers.Dropout(0.4),
		tf.keras.layers.Dense(64, name = "hidden_3"),
		tf.keras.layers.Dense(1, name = "output")
	])
 
	# Compile the model.
	model.compile(
		loss = "mae",
		optimizer = "adam"
	)
 
	return model

def train(model: tf.keras.Model,
          train_data: pd.DataFrame = train_data,
          train_labels: pd.DataFrame = Y,
          epochs: int = 200,
          sample_weights: np.array = sample_weights,
          ) -> None:
	"""
	Trains the TensorFlow `model`.
 
	Parameters
	----------
	model :
		A TensorFlow compiled model.
  
	Returns
	-------
	`None`
	"""
 
	# Check on overfitting.
	early_stopping = tf.keras.callbacks.EarlyStopping(
		monitor =  "val_loss",
		min_delta = 0,
		patience = 100, 
		restore_best_weights = True,
	)

	history = model.fit(
		np.array(train_data).reshape(-1, 1, N_FEATURES), np.array(train_labels),
		sample_weight = sample_weights,
		validation_split = 0.4,
		verbose = 1, 
		epochs = epochs,
		callbacks = early_stopping
	)
	# Loss plots.
	history_df = pd.DataFrame(history.history)
	plt.rcParams["figure.dpi"] = 160
	loss_plot = history_df.loc[:, ["loss", "val_loss"]].plot()
	plt.savefig(os.path.join(os.path.relpath("../plots/training"), ".".join(["training_loss", "png"])))
	print(f"Minimum validation loss: {history_df['val_loss'].min()}")



def main():
	model = build_and_compile()
	train(model)
 
	predictions = model.predict(
		np.array(test_data).reshape(-1, 1, N_FEATURES)
	)
 
	print(pd.DataFrame(np.vstack(
		[predictions.flatten(), 
		Y_test.to_numpy()]).T, 
		columns = ["Predictions", "Real"]))

	print(f"{mean_absolute_error(Y_test, predictions.flatten()) * 100:.2f}")


if __name__ == "__main__":
    main()
