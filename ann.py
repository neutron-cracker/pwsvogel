import os
import data
import pandas
from data import Data_processing
from model import Model as modelClass
import tensorflow
# main functions

# --------------------------------------------------------------------------------------------------------
# Read CSV file and parse dates into pandas dataframe.
# --------------------------------------------------------------------------------------------------------

def get_data(path_to_csv):
    raw_dataframe = pandas.read_csv(path_to_csv)
    raw_dataframe.info()
    raw_weather_dataframe = Data_processing.process_time_data(raw_dataframe)
    clean_dataframe = Data_processing.process_weather_data(
        raw_weather_dataframe)
    transformed_data_frame = data.transform_data_frame.transform_data(
        clean_dataframe)
    #dataset = Data_processing.convert_pandas_dataframe_to_tf_dataset(
        #transformed_data_frame)
    return transformed_data_frame

# --------------------------------------------------------------------------------------------------------
# Variables for DATA-CSV file and get the data.
# --------------------------------------------------------------------------------------------------------

root_path = os.getcwd()
path_to_csv = os.path.join(root_path, "data/weather_data.csv")
weather_data = get_data(path_to_csv)
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
# Define model, set target_data and pecify the amount of epochs
# --------------------------------------------------------------------------------------------------------

length_dataframe = len(weather_data) # count numbers of rows in csv file

train_dataframe = weather_data[0:int(0.7*length_dataframe)] # takes 70% of that number of rows
validation_dataframe = weather_data[int(0.7*length_dataframe):int(0.9*length_dataframe)] # takes the next 20% of that number of rows
test_dataframe = weather_data[int(0.9*length_dataframe):length_dataframe] # takes the last 10% of that number of rows


model = modelClass.build_model()
target_data = "" # TODO
epochs = 2;

modelClass.train_model(model, train_dataframe, target_data, epochs, validation_dataframe)

# --------------------------------------------------------------------------------------------------------
# Read CSV file and parse dates into pandas dataframe.
# --------------------------------------------------------------------------------------------------------

trained_model = modelClass.train_model(model, train_data)
model.save()
