"""
Utility
=======
"""

import numpy as np
import pandas as pd
from typing import List, Iterable

def count_lossy_error(predictions: Iterable, labels: Iterable, tolerance: float = 2) -> float:
	"""
	Compares the `predictions` and `labels` with a `tolerance.`
	
	Parameters
	----------
	predictions :
		An array of predicted values.
  
	labels :
		An array of actual values.
  
	tolerance :
		A number within which the error is acceptable.
  
	Returns
	-------
	float:
		The accuracy of the predictions with tolerance.
	
	Raises
	------
	AttributeError:
		If predictions and labels are not of the same length.
	"""
	if len(predictions) != len(labels):
		raise ValueError(f"The shapes of prediction and labels should match in dim 0.")
	
	return sum(1 for pred, label in zip(predictions, labels) if abs(pred - label) >= tolerance) / len(predictions)

def scale(data: pd.DataFrame, cols: List[str] = None, type: str = "mean") -> pd.DataFrame:
	"""
	Normalises columns of `data` using the method `type`.
	
	Parameters
	----------
	data :
		The dataframe who's columns need to be normalised.
  
	cols : 
		The columnss to be normalised. By deafult is `None` which mean normalises the entire dataframe.
	
	type : 
		Acceptable methods are *mean* and *min-max*.

	Returns
	-------
	pd.DataFrame:
		A normalised version of the input dataframe.
	"""
	norm_data = data.copy()
 
	if cols is None:
		return (norm_data - norm_data.mean()) / norm_data.std()

	if type == "mean":
		norm_data[cols] = (norm_data[cols] - norm_data[cols].mean()) / norm_data[cols].std()

	if type == "min-max":
		norm_data[cols] = (norm_data[cols] - norm_data[cols].min()) / (norm_data[cols].max() - norm_data[cols].min())
  
	return norm_data
	