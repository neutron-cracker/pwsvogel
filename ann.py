import os

from pandas.core.frame import DataFrame
import data
import pandas
from data import Convert_data
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

weather_data = data.Get_data.weather_data(path_to_csv_weather)
bird_data = data.Get_data.bird_data(path_to_csv_bird)
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

train_bird_dataframe = bird_data[0:int(0.7*length_dataframe)]
validation_bird_dataframe = bird_data[int(
    0.7*length_dataframe):int(0.9*length_dataframe)]
test_bird_dataframe = bird_data[int(0.9*length_dataframe):length_dataframe]


# target_data = Data_processing.process_bird_migration_data(bird_data)  # TODO
train_dataset = Convert_data.dataframe_to_tf_dataset(
    train_weather_dataframe, train_bird_dataframe)  # , train_dataframe)
epochs = 50

# export dataframe


def exportFile(dataset, fileName):
    csv = pandas.DataFrame.to_csv(dataset)
    outfileName = os.path.join(root_path, fileName)
    outFile = open(outfileName, "w")
    outFile.write(csv)
    outFile.close()


train_bird_dataframe.drop(train_bird_dataframe.index[0])
exportFile(train_bird_dataframe, "bird.csv")
exportFile(train_weather_dataframe, "weather.csv")


# --------------------------------------------------------------------------------------------------------
# Train model and save the model.
# --------------------------------------------------------------------------------------------------------

model.fit(x=train_weather_dataframe, y=train_bird_dataframe, epochs=epochs, verbose=2, validation_data = (validation_weather_dataframe, validation_bird_dataframe))
model.save(os.path.join(root_path, "pwsvogelmodel"))
predictions = model.predict(x=test_weather_dataframe)
predictions = pandas.DataFrame(data=predictions)
exportFile(predictions, "predictions.csv")
