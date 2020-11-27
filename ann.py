import os
import data
import pandas
from data import Data_processing
from model import Model as modelClass
import tensorflow
import numpy
# main functions

# --------------------------------------------------------------------------------------------------------
# Read CSV file for weather and parse dates into pandas dataframe.
# --------------------------------------------------------------------------------------------------------

root_path = os.getcwd()
path_to_csv_weather = os.path.join(root_path, "data/weather_data.csv")
path_to_csv_bird = os.path.join(
    root_path, "data/bird_migration_per_specie.csv")

weather_data = data.Get_data.get_data_weather(path_to_csv_weather)
bird_data = data.Get_data.get_data_bird(path_to_csv_bird)
# --------------------------------------------------------------------------------------------------------
# Call : Build the model.
# --------------------------------------------------------------------------------------------------------

model = modelClass.build_model()

# --------------------------------------------------------------------------------------------------------
# Get number of rows and calculate the amount needed for respectively train, validation, and test data.
# Define model, set target_data and pecify the amount of epochs.
# --------------------------------------------------------------------------------------------------------

length_dataframe = len(weather_data)  # count numbers of rows in csv file

# takes 70% of that number of rows
train_weather_dataframe = weather_data[0:int(0.7*length_dataframe)]
# takes the next 20% of that number of rows
validation_weather_dataframe = weather_data[int(
    0.7*length_dataframe):int(0.9*length_dataframe)]
# takes the last 10% of that number of rows
test_weather_dataframe = weather_data[int(
    0.9*length_dataframe):length_dataframe]

train_bird_dataframe = weather_data[0:int(0.7*length_dataframe)]
validation_bird_dataframe = weather_data[int(
    0.7*length_dataframe):int(0.9*length_dataframe)]
test_bird_dataframe = weather_data[int(0.9*length_dataframe):length_dataframe]


#target_data = Data_processing.process_bird_migration_data(bird_data)  # TODO
train_dataset = Data_processing.convert_pandas_dataframe_to_tf_dataset(
    train_weather_dataframe, train_bird_dataframe)  # , train_dataframe)
epochs = 2
model.fit(x=train_weather_dataframe, y=train_bird_dataframe, epochs=epochs)
modelClass.train_model(model, train_dataset,
    epochs)

# --------------------------------------------------------------------------------------------------------
# Give command to train model and save the model.
# --------------------------------------------------------------------------------------------------------

trained_model = modelClass.train_model(model, train_dataset)
model.save()
