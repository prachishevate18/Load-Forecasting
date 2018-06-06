from pandas import Series
import csv
import pandas as pd
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

#read the file and calculate the average per day
series = Series.from_csv('load_temperature_data.csv',header=0)
print(series.count())
resample = series.resample('87300s')
daily_mean = resample.mean()
cnt = daily_mean.count()
df = pd.read_csv('load_temperature_data.csv').fillna(value=0)
df = df[["timestamps","actual_temperature"]]

#remove the unnecessary date formats
df1 = df["timestamps"].str.split('-07:00').str[0]
df2 = df["actual_temperature"]
result = pd.concat([df1,df2], axis=1)

df3 = result["timestamps"].str.split('-08:00').str[0]
df4 = result["actual_temperature"]
result1 = pd.concat([df3,df4], axis=1)
result1.to_csv('output1.csv',index=False)


#count the number of rows in CSV file
input_file = open("load_temperature_data.csv","r+")
reader_file = csv.reader(input_file)
for row in reader_file:
    value = len(list(reader_file))
print(value)

#To calculate the average for missing values
j = 0
i = 0
no_of_records=96 #number of records per day
k = 96
while(j < no_of_records):
    if(j==value):
        break
    get_val = result1.get_value(j,"actual_temperature")
    if(get_val == 0.0):
        result1.set_value(j,"actual_temperature",daily_mean[i])
    if(j == no_of_records - 1):
        i = i + 1
        if(i == cnt):
            break
            no_of_records = k + no_of_records
    j = j + 1


result1.to_csv('output1.csv',index=False)

#plot to show the first average value calculations
result1.plot()
pyplot.show()

# To calculate the average per month
series1 = Series.from_csv('output1.csv',parse_dates=True)
resample = series.resample('M')
monthly_mean = resample.mean()
cnt = monthly_mean.count()
print(monthly_mean.head())


j = 0
i = 0
month_by_day= 2880  #product of number of days in a month and number of records per day
k = 2880
while(j < month_by_day):
    if(j == value):
        break
    get_val = result1.get_value(j,"actual_temperature")
    result1.set_value(j,"actual_temperature",get_val - monthly_mean[i])
    if(j == month_by_day - 1):
        i = i + 1
        if(i == cnt):
            break
            month_by_day = k + month_by_day + 97
    j = j + 1
result1.to_csv('output2.csv',index=False)
#plot to show the final average value calculations
result1.plot()
pyplot.show()

#ARIMA model for Forecast and prediction of Temperatures
X = result1["actual_temperature"].values
size = 45505
#train, test = X[1:5000], X[5000:6000]
train, test = X[0:30000], X[30000:size]
history = [x for x in train]
predictions = list()
# walk-forward validation
for t in range(len(test)):
	# fit model
    model = ARIMA(history, order=(2,1,1))
    model_fit = model.fit(disp=False)
    # one step forecast
    yhat = model_fit.forecast()[0]
    # store forecast and ob
    predictions.append(yhat)
    history.append(test[t])
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Predicted value: %.3f' % rmse)