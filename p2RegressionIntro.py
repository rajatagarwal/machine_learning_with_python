import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression


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
# we'll fill some number which makes that value an outlier

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

# we will be applying linear regression to our data
# we've X and y. X is the feature and y is the label

X = np.array(df.drop(["label"], 1))
y = np.array(df["label"])

# NOTE: we can scale X before feeding it to classifier. It normalize it.
# It is not mandatory. So if you're using it, you always need to scale
# it with new values. (Not only new values, but along with old values)

X = preprocessing.scale(X)

# Here I used to write cross_validation.train_test_split (After import cross_validation)
# but that's now depreciated.
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# Let's create an object of our classifier
clf_linear_regression = LinearRegression()

# Now let's use this classifier with our training data
clf_linear_regression.fit(X_train, y_train)

# Also, we need to evaluate our model with the test data.
# fit is synonyms with train and score is synonyms with test data
# We need to train and test classifier on the separate data because...\
# Also, in linear regression an accuracy is squared error.
accuracy_linear_regression = clf_linear_regression.score(X_test, y_test)

print("Linear Regression Accuracy", accuracy_linear_regression)

# We can create other models as well and test those
# lets do with Scalar Vector Regression

clf_svr = svm.SVR()
clf_svr.fit(X_train, y_train)
accuracy_svr = clf_svr.score(X_test, y_test)
print("SVR Accuracy", accuracy_svr)

# We can also change kernel in SVR(and also in some other ML Algo) and set to "poly"
clf_svr_poly = svm.SVR(kernel="poly")
clf_svr_poly.fit(X_train, y_train)
accuracy_svr_poly = clf_svr_poly.score(X_test, y_test)
print("SVR poly accuracy", accuracy_svr_poly)

# By default linear regression runs single thread at a time.
# we can change number of threads to make parallel jobs by passing n_jobs value

# below model run with 10 threads parallel
clf_lr_multiple_threads = LinearRegression(n_jobs=10)
clf_lr_multiple_threads.fit(X_train, y_train)
# i think accuracy shouldn't change just because we're running parallel processes.
accuracy_lr_multiple_threads = clf_lr_multiple_threads.score(X_test, y_test)
print("Parallel threads linear regression accuracy", accuracy_lr_multiple_threads)
# I was right, accuracy didn't change. But process was quite fast. A good tip.

# you can also give n_jobs = -1, then it will run as many jobs as manage
# by your processor
clf_lr_max_threads = LinearRegression(n_jobs=-1)
clf_lr_max_threads.fit(X_train, y_train)
accuracy_lr_max_threads = clf_lr_max_threads.score(X_test, y_test)
# accuracy should be same but it will run ultra fast. Mac Rocks. <3 :P
print("Max threads linear regression accuracy", accuracy_lr_max_threads)