import os
from typing import Union

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import utility as ut
from variables import *


# Read the data.
train_data = pd.read_csv(os.path.join(DATA_PATH, ".".join([DATA_TRAIN, DATA_EXT])), header = 0)

# Get the labels.
Y = train_data.pop(LABEL)

sample_weights = np.ones(Y.shape[0])
for i in range(10, 24):
	sample_weights[train_data["_".join(("hour", str(i)))] == 1] = 1.5
 
# CLASSES = np.unique(Y)
# N_CLASSES = len(CLASSES)
# Y = Y.replace(dict(zip(CLASSES, range(0, len(CLASSES)))))
 

# Data shape parameters.
N_FEATURES = train_data.shape[1]
N_SAMPLES = train_data.shape[0]

# Split the training data.
X_train, X_val, Y_train, Y_val = train_test_split(train_data, Y, shuffle = True, random_state = 7919)



def build_and_compile(input_: tuple = (WB_SIZE, N_FEATURES)
                      , loss_func: str = "mae") -> tf.keras.Model:
	"""
	Build and compile a TensorFLow LSTM network.
	
	Parameters
	----------
	input_ :
		Shape of the trainining data. Should specify 
		`(batch_size` or `window_size, n_features)`
	loss_func :
		Loss function to use for training.
  
	Returns
	-------
	`tf.keras.Model` :
		A compiled TensorFlow model.
	"""
 
	# Seqential keras model.
	model = tf.keras.models.Sequential([
		tf.keras.layers.LSTM(50, input_shape = input_, return_sequences = True),
		# tf.keras.layers.Dropout(0.4),
		tf.keras.layers.LSTM(50, return_sequences = False),
		tf.keras.layers.GaussianNoise(0.5),
		tf.keras.layers.Dense(512),
		# tf.keras.layers.Dropout(0.7),
		tf.keras.layers.Dense(128),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(64),
		# tf.keras.layers.Dropout(0.7),
		tf.keras.layers.Dense(1, activation = "linear")
	])
 
	# Compile the model.
	model.compile(
		loss = loss_func,
		optimizer = "adam"
	)
 
	return model

def train(model: tf.keras.Model,
		  train_data: pd.DataFrame = X_train,
		  train_labels: pd.DataFrame = Y_train,
		  val_data: pd.DataFrame = X_val,
		  val_labels: pd.DataFrame = Y_val,
		  epochs: int = 300,
		  sample_weights: np.array = None,
		  ) -> pd.DataFrame:
	"""
	Trains the TensorFlow `model`.
 
	Parameters
	----------
	model :
		A TensorFlow compiled model.
	train_data :
		The data to be trained. Shape must be consistent with what is passed during model compilation.
	train_labels :
		The ground truth predictions.
	val_data :
		The data to be used as validation.
	val_labels : 
		The ground truth validation predictions.
	epochs :
		Total number of epochs to train.
	sample_weights :
		Weights for `train_data` to use during training.
  
	Returns
	-------
	pd.DataFrame:
		Training information.
	"""
 
	# Check for overfitting.
	early_stopping = tf.keras.callbacks.EarlyStopping(
		monitor =  "val_loss",
		min_delta = 0,
		patience = 100, 
		restore_best_weights = False,
	)

	history = model.fit(
		np.array(train_data).reshape(-1, WB_SIZE, N_FEATURES), 
  		np.array(train_labels),
		validation_data = (np.array(val_data).reshape(-1, WB_SIZE, N_FEATURES), np.array(val_labels)),
		verbose = 1, 
		epochs = epochs,
		callbacks = early_stopping
	)
 
	return pd.DataFrame(history.history) 
	

def train_stats(history_df: pd.DataFrame) -> None:
	"""
	Produces training statistics once it training has run its course.
	
	Parameters
	----------
	history_df :
		The history as returned by the `train` method.
  
	Returns
	-------
	`None`
	
	"""
	print("Training complete.")
	# Learning curve.
	plt.rcParams["figure.dpi"] = 160
	history_df.loc[:, ["loss", "val_loss"]].plot()
	plt.title("Model Loss")
	plt.ylabel("Loss")
	plt.xlabel("Epoch")
	plt.savefig(os.path.join(TRAIN_FIG_SAVE_PATH, ".".join(["learning_curves", FIG_EXT])))
 
	# Stats 
	print(f"Minimum validation loss: {history_df['val_loss'].min()}")
	# plt.plot(f"Accuracy: {history_df['train_accuracy']}")
	# plt.plot(f"Validation Accuracy: {history_df['val_accuracy']}")
 
	return None

def main():
	model = build_and_compile((WB_SIZE, N_FEATURES), "mae")
	history_df = train(model)
	
	train_stats(history_df)
 
	# Save trained model (better to use checkpoints).
	model.save(os.path.join(NN_MODEL_SAVE_PATH, NN_MODEL_SAVE_NAME))


if __name__ == "__main__":
	main()
