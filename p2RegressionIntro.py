import pandas as pd
import quandl
import math

# DAY 1
quandl.ApiConfig.api_key = 'zbd-9gSkWFZGDmWDNTGa'

df = quandl.get_table('WIKI/PRICES')

# We just need adj. columns for our regression model. 

df = df[["adj_open", "adj_high", "adj_low", "adj_close", "adj_volume"]]

# Let's create a new column which captures variation between high and low. We'll calculate the percentage change

df["HL_PC"] = (df["adj_high"] - df["adj_low"]) / df["adj_low"] * 100

# We can also find rates change on daily basis - close and open.

df["CO_PC"] = (df["adj_close"] - df["adj_open"]) / df["adj_open"] * 100

# So for further analysis we need following columns only

df = df[["adj_close", "HL_PC", "CO_PC", "adj_volume"]]

# DAY 2

# above all are the features. we need to find a label. numeric column like hl_pc and co_pc are not labels 
# and the only column related to money is adj_close. So we'll be forecasting on adj_close

forecast_col = "adj_close"

# let's fill all NA values with some arbitrary number. Machine learning algorithm doesn't work will null values.
# we'll fill some number wihch makes that value an outlier

df.fillna(-99999, inplace=True)

# Lets create a label out of adj_close. Here we're taking 0.1 which is by choice. You can take any other number.
forecast_out = int(math.ceil(.001 * len(df)))


# len(df) tells us there are 9900 items in the dataframe. so as per the above formula
# forecast_out = int(math.ceil(.001 * 9900))
# forecast_out = int(math.ceil(9.9))
# forecast_out = 10
# so we will be forecasting for 10 days. We're shifting each data by 10 days 
# more about shift operation here:
# https://stackoverflow.com/questions/20095673/python-shift-column-in-pandas-dataframe-up-by-one
df["label"] = df[forecast_col].shift(-forecast_out)

# if you run a below command now, you would see last 10 values in label column as NaN.
# That's because we shifted 10 values up.
# print(df.tail(20))
# to get rid of these NaN we can drop na
df.dropna(inplace=True)

# now our data is ready.
print(df.head())
print(df.tail())

# Day 3

