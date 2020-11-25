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


def get_data_weather(path_to_csv):
    raw_dataframe = pandas.read_csv(path_to_csv)
    raw_dataframe.info()
    raw_weather_dataframe = Data_processing.process_time_data(raw_dataframe)
    clean_dataframe = Data_processing.process_weather_data(
        raw_weather_dataframe)
    transformed_data_frame = data.transform_data_frame.transform_data(
        clean_dataframe)
    # dataset = Data_processing.convert_pandas_dataframe_to_tf_dataset(
    # transformed_data_frame)
    return transformed_data_frame

# --------------------------------------------------------------------------------------------------------
# Read CSV file for birds and parse dates into pandas dataframe.
# --------------------------------------------------------------------------------------------------------


def get_data_bird(path_to_csv):
    raw_dataframe = pandas.read_csv(path_to_csv)
    raw_dataframe.info()
    raw_bird_dataframe = Data_processing.process_time_data(raw_dataframe)
    # dataset = Data_processing.convert_pandas_dataframe_to_tf_dataset(
    # transformed_data_frame)
    return raw_bird_dataframe

# --------------------------------------------------------------------------------------------------------
# Variables for DATA-CSV files and get the data.
# --------------------------------------------------------------------------------------------------------


root_path = os.getcwd()

path_to_csv_weather = os.path.join(root_path, "data/weather_data.csv")
path_to_csv_bird = os.path.join(root_path, "data/bird_migration.csv")

weather_data = get_data_weather(path_to_csv_weather)
bird_data = get_data_bird(path_to_csv_bird)

# TODO type thing to split data

# --------------------------------------------------------------------------------------------------------
# Call : Build the model.
# --------------------------------------------------------------------------------------------------------

model = modelClass.build_model()
tensorflow.keras.utils.plot_model(  # can eventually be removed
    model, to_file='model.png'
)

# --------------------------------------------------------------------------------------------------------
# Get number of rows and calculate the amount needed for respectively train, validation, and test data.
# Define model, set target_data and pecify the amount of epochs.
# --------------------------------------------------------------------------------------------------------

length_dataframe = len(weather_data)  # count numbers of rows in csv file

# takes 70% of that number of rows
train_dataframe = weather_data[0:int(0.7*length_dataframe)]
# takes the next 20% of that number of rows
validation_dataframe = weather_data[int(
    0.7*length_dataframe):int(0.9*length_dataframe)]
# takes the last 10% of that number of rows
test_dataframe = weather_data[int(0.9*length_dataframe):length_dataframe]


target_data = Data_processing.process_birdMigration_data(bird_data)  # TODO
epochs = 2
train_dataset = Data_processing.convert_pandas_dataframe_to_tf_dataset(
    train_dataframe, bird_data)  # , train_dataframe)
print(train_dataframe)
modelClass.train_model(model, train_dataset, target_data,
                       epochs, validation_dataframe)

# --------------------------------------------------------------------------------------------------------
# Give command to train model and save the model.
# --------------------------------------------------------------------------------------------------------

trained_model = modelClass.train_model(model, train_data)
model.save()
