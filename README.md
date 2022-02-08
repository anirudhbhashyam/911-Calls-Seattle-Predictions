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

# Running The Model
```
git clone --recursive https://github.com/anirudhbhashyam/911-Calls-Seattle-Predictions
cd src
```
## Training
Pretrained models are saved in *models*. To train the neural netwok, one can do the following

```
python train_nn.py
```
Training will print trainining statistics and save the learning curve to `./plots/training`.

## Testing
Trained models that are saved in *./models* are automatically loaded and tested when 
```
python test.py
```
is called. Testing prints testing statistics and saves backtest plots to `./plots/testing`. 

# Documentation
The documentation is built using sphinx but is not published yet. 

