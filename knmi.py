
import os
from  datetime import datetime, timedelta
from pandas import DataFrame
from tensorflow.keras import  models
from os import  path, getcwd
from data import Get_data
from pastas.read.knmi import KnmiStation

date_of_today = datetime.now().date()
date_of_5_days_back = date_of_today - timedelta(5)
knmi = KnmiStation.download(start=date_of_5_days_back, end=date_of_today)
knmi_data = knmi.data
arrayWithVariables = ['FHVEC', 'DDVEC', 'TG', 'RH']

def getNeededVariables(arrayWithVariables):
    path_to_model = path.join(getcwd(), 'pwsvogelmodel')
    neededKNMIdata = DataFrame()
    for variable in arrayWithVariables:
        neededKNMIdata[variable] = knmi_data.pop(variable)

    neededKNMIdata = neededKNMIdata.reset_index()
    neededKNMIdata.rename(
        columns={neededKNMIdata.columns[0]: "DATE"}, inplace=True)

    neededKNMIdata = Get_data.transform_weather_data(
        neededKNMIdata).drop('DATE', axis=1)
    model = models.load_model(path_to_model)
    print(model.summary())
    predictions = model.predict(x=neededKNMIdata)
    Get_data.exportFile(DataFrame(predictions), 'predictions.csv')
    return predictions

prediction_based_on_knmi_date = getNeededVariables(arrayWithVariables)
print(prediction_based_on_knmi_date)
