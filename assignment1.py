#install prophet
#%pip install prophet
#import libraries
import pandas as pd
import numpy as np
from prophet import Prophet

#ingest data
train_url ="https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv"
test_url = "https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_test.csv"

train = pd.read_csv(train_url)
test = pd.read_csv(test_url)
train = train[['Timestamp', 'trips']].rename(columns={'Timestamp': 'ds', 'trips': 'y'}).assign(ds=lambda df: pd.to_datetime(df['ds']))
train

#creating and fitting the model
model = Prophet(changepoint_prior_scale=0.5)
modelFit = model.fit(train)

# creating timeline for 744 periods in the future (January 20xx) and generating predictions
future = model.make_future_dataframe(periods=744)
pred = model.predict(future)

#visualizing the forecast and components
plt = model.plot(pred)
comp = model.plot_components(pred)