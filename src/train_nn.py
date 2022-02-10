import os
from typing import Union

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold

import utility as ut
from variables import *


# Read the data.
train_data = pd.read_csv(os.path.join(DATA_PATH, ".".join([DATA_TRAIN, DATA_EXT])), header = 0)

# Get the labels.
Y = train_data.pop(LABEL)

sample_weights = np.ones(Y.shape[0])
for i in range(10, 24):
	sample_weights[train_data["_".join(("hour", str(i)))] == 1] = 1.5


#  -- For classification -- #
# CLASSES = np.unique(Y)
# N_CLASSES = len(CLASSES)
# Y = Y.replace(dict(zip(CLASSES, range(0, len(CLASSES)))))
 

# Data shape parameters.
N_FEATURES = train_data.shape[1]
N_SAMPLES = train_data.shape[0]

# Split the training data.
X_train, X_val, Y_train, Y_val = train_test_split(train_data, Y, shuffle = True, random_state = 7919)

def build_and_compile(input_: tuple = (WB_SIZE, N_FEATURES), 
                      loss_func: str = "mae") -> tf.keras.Model:
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
		tf.keras.layers.LSTM(50, return_sequences = False),
		tf.keras.layers.GaussianNoise(1.0),
		tf.keras.layers.Dense(1024, activation = "relu"),
		tf.keras.layers.Dropout(0.7),
		tf.keras.layers.Dense(128, activation = "relu"),
		tf.keras.layers.Dropout(0.2),
		tf.keras.layers.Dense(64, activation = "relu"),
		tf.keras.layers.GaussianNoise(0.2),
		# tf.keras.layers.Dense(32, activation = "relu"),
		# tf.keras.layers.GaussianNoise(0.7),
		tf.keras.layers.Dense(1, activation = "relu")
	])
 
	# Compile the model.
	model.compile(
		loss = loss_func,
		optimizer = "adam"
	)
 
	return model

def train(model: tf.keras.Model,
          train_data: np.ndarray,
          train_labels: np.ndarray,
          val_data: np.ndarray,
          val_labels: np.ndarray,
		  epochs: int = 200,
		  sample_weights: np.array = None,
		  cross_val = False) -> pd.DataFrame:
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
		min_delta = 0.001,
		patience = 100, 
		restore_best_weights = False)
 
	history = model.fit(
		train_data.reshape(-1, WB_SIZE, N_FEATURES), 
  		train_labels,
		sample_weight = sample_weights,
		validation_data = (val_data.reshape(-1, WB_SIZE, N_FEATURES), val_labels),
		verbose = 1, 
		epochs = epochs,
		callbacks = early_stopping)
 
	return pd.DataFrame(history.history)

# def cross_validate(train_data: pd.DataFrame,
#                    train_labels: pd.DataFrame,
#                    epochs: int = 50,
#                    sample_weights: np.array = None,
#                    folds: int = 2) -> pd.DataFrame:
    
# 	splits = KFold(n_splits = folds, shuffle = True)
# 	print("Starting cross validation.")
# 	accuracy = list()
# 	val_loss = list()
# 	models = list()
# 	for i, (train_index, test_index) in enumerate(splits.split(train_data, train_labels)):
# 		print(f"Iteration {i}\n")
# 		X_train, X_val, Y_train, Y_val = train_data[train_index], train_data[test_index], train_data[train_index], train_labels[test_index]
  
# 		model = build_and_compile((WB_SIZE, N_FEATURES), "mae")
  
# 		history_df = train(model, X_train, Y_train, epochs)
# 		# train_stats(history_df, i)
		
# 		scores = model.evaluate(X_val.reshape(-1, WB_SIZE, N_FEATURES), Y_val)
# 		print(f"Validation loss: {scores}\n")
#         #of {scores[0]} {model.metrics_names[1]} of {scores[1] * 100:.2f}%")

# 		# accuracy.append(scores[1] * 100)
# 		val_loss.append(scores)
# 		models.append(model)
  
# 	return models[np.argmin(val_loss)]
	 
def train_stats(history_df: pd.DataFrame, it: int = None) -> None:
	"""
	Produces training statistics once training has run its course.
	
	Parameters
	----------
	history_df :
		The history as returned by Keras `fit` method.
	it :
		To be used with cross validation. Specifies the name of the learning curve based on the cross validation itertation `it`. 
  
	Returns
	-------
	`None`
	
	"""
	# Learning curve.
	plt.rcParams["figure.dpi"] = 160
	history_df.loc[:, ["loss", "val_loss"]].plot()
	plt.title("Model Loss")
	plt.ylabel("Loss")
	plt.xlabel("Epoch")
	name = TRAIN_FIG_SAVE_NAME
	if it is not None:
		name = "_".join([name, str(it)])
	plt.savefig(os.path.join(TRAIN_FIG_SAVE_PATH, ".".join([name, FIG_EXT])))
 
	# Stats 
	print(f"Minimum validation loss: {history_df['val_loss'].min()}")
	# plt.plot(f"Accuracy: {history_df['train_accuracy']}")
	# plt.plot(f"Validation Accuracy: {history_df['val_accuracy']}")
 
	return None

def main():
	model = build_and_compile((WB_SIZE, N_FEATURES))
	# model = cross_validate(np.array(train_data), np.array(Y))
	history_df = train(model, np.array(X_train), np.array(Y_train), np.array(X_val), np.array(Y_val))
	
	# train_stats(history_df)
 
	# Save trained model (better to use checkpoints).
	model.save(os.path.join(NN_MODEL_SAVE_PATH, NN_MODEL_SAVE_NAME))

if __name__ == "__main__":
	main()
