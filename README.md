# 911 Calls Seattle Prediction
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Prerequisites
Running the prediction model requires these basic dependancies:

* Python `>=3.8.0`
* TensorFlow `>=2.7.0`
* scikit-learn `>=0.24.1` 
* Sphinx (for building documentation) `>=4.3.0`.

Detailed environment information can be found in `requirements.txt`.


# Data 
Data was procured from [source](https://data.seattle.gov/Public-Safety/Seattle-Real-Time-Fire-911-Calls/kzjm-xkqj). It was handled initially with `SQL`. Using `SQL`, the number of records was reduced to the last 7 years and the *Datetime* was split into *Date* and *Time*. Subsequently `pandas` was used to clean and transform the data. `clean.ipynb` details how the data was cleaned, calls extracted and  prepared for transformation. `transform.ipynb` scales the data and adds new features based on the old features.

To run the notebooks, the data must be downloaded from the *source*, the preprocessing directory can be used to query the data to create the reduced data file like so 
```
cd data
mkdir raw
```
query data using preprocessing files
Save queried data to raw as `raw_reduced_7_year.csv` 

# Running The Model
```
git clone --recursive https://github.com/anirudhbhashyam/911-Calls-Seattle-Predictions
cd src
```
## Training
Pretrained models are saved in *models*. To train the neural netwok, one can do the following

```
python train_nn.py
python train_gb.py
```
Training will print training statistics and save the learning curve to `./plots/training`.

## Testing
Trained models that are saved in *./models* are automatically loaded and tested when 
```
python test.py
```
is called. Testing prints testing statistics and saves backtest plots to `./plots/testing`. Testing with the sklearn model is not possible on a pretrained model.


# Documentation
The documentation is built using sphinx but is not published yet. Current version can be used in `/docs/_build/html/index.html`.

# Improvements
* OOP approach with TensorFlow network. This will allow more flexibility in the architecture and functionality in the network (for example classification can be done).
* Testing the network using cross validation and more hyperparameters.
* Hyperparameter estimation in scikit-learn estimator.
* Better documentation.
* Add unittests for network and other functions.
* Rethink feature space in the dataset. 
