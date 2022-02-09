# In[53]:

import os

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

print("Setup complete.")


# In[54]:


path = os.path.abspath("./raw")
file = "raw_reduced_7_years"
ext = "csv"

raw_data = pd.read_csv(os.path.join(path, ".".join([file, ext])))
print("Raw data loaded.")


# In[55]:


# Convert features to datetime and get day of year.
raw_data["Date"] = pd.to_datetime(raw_data["Date"])
raw_data["Time"] = pd.to_datetime(raw_data["Time"], format = "%H:%M:%S")
raw_data["day_of_year"] = raw_data["Date"].dt.day_of_year
raw_data["hour"] = raw_data["Time"].dt.hour
print("Converted datetime features to correct format.")
print("Added day_of_year and hour as features.") 


# In[57]:


# Drop unnecessary columns.
columns_to_drop = ["Incident_Number", "Report_Location", "Address", "Type"]
data = raw_data.drop(columns = columns_to_drop)
data = data.sort_values(by = "Date")
# data[["Time", "Date"]] = data[["Time", "Date"]].isna()
print(f"Dropped columns: {columns_to_drop}")

# In[58]:

action = "impute"

if action == "all":
	data.dropna(inplace = True)
elif action == "impute":
	columns_to_impute = list(data.isna().any()[data.isna().any()].index)
	imp = IterativeImputer(max_iter = 20, random_state = 26)
	imp.fit(data[columns_to_impute])
	new_data = pd.DataFrame(imp.transform(data[columns_to_impute]), columns = columns_to_impute)
	data.drop(columns = columns_to_impute)
	data[columns_to_impute] = new_data
elif action == "mean replace":
	data_filled = data.fillna(data.mean())
else:
	print("Select specific action.")

print(f"Handled missing values using {action}.")
	
def origin_haversine(coord: tuple, degrees = True) -> float:
    """
    Calculates the Haversine the point `(latitude, longitude)` and `(0, 0)`.
    
    Parameters
    ----------
    coord:
        The coordinates specified as `(latitude, longitude)` either in degrees or radians.
    degrees:
        If true converts coordinates from (assumed) degrees to radians.
        
    Returns
    -------
    float:
        The distance.
    """
    lat, lng = coord
    
	# Earth's radius in km.
    r = 6371 

    # Convert decimal degrees to radians, if needed.
    if degrees:
        lat, lng = map(np.radians, [lat, lng])

    # Harvesine distance between (0, 0) and (lat, long)
    a = np.sin(lat / 2) ** 2 + np.cos(lat) * np.sin(lng / 2) ** 2
    d = 2 * r * np.arcsin(np.sqrt(a)) 

    return d

HAVERSINE_FEATURE = False

if HAVERSINE_FEATURE:
	print("Added Haversine distance as feature.")
	data["latlong_combined"] = [origin_haversine((lat, lng)) for lat, lng in zip(data.Latitude, data.Longitude)]
	data_haversine = data.drop(columns = ["Latitude", "Longitude"])


# In[71]:


def create_main_data(data: pd.DataFrame, date: str, time_groups: list):
	"""
	Splits the dataframe `data` by year using the date column `date`. Groups by the specified time groups (eg: `year_of_day` and `hour`), averages the latitudes and longitudes.
	
	Parameters
	----------
	data:
		The data to be processed.
  
	date:
		Column name of the date column in `data`.
  
	Returns
	-------
	list:
		A list of dataframes.
	"""
	year_frames = [data[data[date].dt.year == y] for y in data[date].dt.year.unique()]
	main_frames = list()
	for df in year_frames:
		d_temp = df.groupby(time_groups)			.agg({
					date: ["count"],
					"Latitude": ["mean"],
					"Longitude": ["mean"]
			})\
			.reset_index(time_groups)\
			.sort_values(time_groups)
   
		d_temp.rename(columns = {date: "calls"}, inplace = True)
		d_temp.columns = d_temp.columns.droplevel(1)
		main_frames.append(d_temp)
	return main_frames


# In[72]:
main_frames = create_main_data(data, "Date", ["day_of_year", "hour"])
print("Created yearly data frames.")

# In[73]:


final_data_mean = pd.concat(main_frames[::-1], ignore_index = True)
final_data_test_mean = main_frames[-1]
print("Created raw train and test data.")

# In[74]:


path_to_save = os.path.relpath("./train-test")
final_data_mean.to_csv(os.path.join(path_to_save, ".".join(["data_yearly_hourly_train", "csv"])), index = False)
final_data_test_mean.to_csv(os.path.join(path_to_save, ".".join(["data_yearly_hourly_test", "csv"])), index = False)
print(f"Saved raw train and test data to {path_to_save}.")

