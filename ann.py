import os
import tensorflow
import numpy
import matplotlib.pyplot as plt
import data
import pandas
from os import path, getcwd()
from pandas.core.frame import DataFrame
from six import print_
from data import Convert_data, Get_data
from model import Model as modelClass
from tensorflow import keras
# --------------------------------------------------------------------------------------------------------
# Read CSV file for weather and parse dates into pandas dataframe.
# --------------------------------------------------------------------------------------------------------

root_path = getcwd()
path_to_csv_weather = path.join(root_path, "data/weather_data.csv")
path_to_csv_bird = path.join(
    root_path, "data/bird_migration_per_specie.csv")

weather_data_raw = Get_data.raw_weather_data(path_to_csv_weather)
bird_data_unfiltered = Get_data.bird_data(path_to_csv_bird)


weather_data_raw = Get_data.transform_weather_data(weather_data_raw)

weather_data = Get_data.filtered_weather_data(
    weather_data_raw, bird_data_unfiltered)
weather_data = weather_data.drop(['DATE'], axis=1)
bird_data = bird_data_unfiltered.drop('DATE', axis=1)


# --------------------------------------------------------------------------------------------------------
# Call : Build the model.
# --------------------------------------------------------------------------------------------------------

model = modelClass.build_model()

# --------------------------------------------------------------------------------------------------------
# Get number of rows and calculate the amount needed for respectively train, validation, and test data.
# Define model, set target_data and pecify the amount of epochs.
# --------------------------------------------------------------------------------------------------------
train_weather_dataframe = Get_data.part_of_data(weather_data, train)
validation_weather_dataframe = Get_data.part_of_data(weather_data, validation)
test_weather_dataframe = Get_data.part_of_data(weather_data, test)

train_bird_dataframe = Get_data.part_of_data(bird_data, train)
validation_bird_dataframe = Get_data.part_of_data(bird_data, validation)
test_bird_dataframe = Get_data.part_of_data(bird_data, test)

for column in range((len(train_bird_dataframe.columns) - 1), 10, -1):
    train_bird_dataframe = train_bird_dataframe.drop(
        train_bird_dataframe.columns[column], axis=1)
for column in range((len(validation_bird_dataframe.columns) - 1), 10, -1):
    validation_bird_dataframe = validation_bird_dataframe.drop(
        validation_bird_dataframe.columns[column], axis=1)
for column in range((len(test_bird_dataframe.columns) - 1), 10, -1):
    test_bird_dataframe = test_bird_dataframe.drop(
        test_bird_dataframe.columns[column], axis=1)

# export dataframe

Get_data.exportFile(train_bird_dataframe, "bird.csv")
Get_data.exportFile(train_weather_dataframe, "weather.csv")

# --------------------------------------------------------------------------------------------------------
# Train model and save the model.
# --------------------------------------------------------------------------------------------------------
   
epochs = 200

history = model.fit(x=train_weather_dataframe, y=train_bird_dataframe, epochs=epochs, verbose=2, shuffle=True,
                    validation_data=(validation_weather_dataframe, validation_bird_dataframe))

training_loss = history.history['loss']
validation_loss = history.history['val_loss']
epoch_count = range(1, len(training_loss)+1)

plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, validation_loss, 'b-')
plt.legend(['Training Error', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss.png')
plt.show()

model.save(path.join(root_path, "pwsvogelmodel"))
predictions = model.predict(x=test_weather_dataframe)
predictions = DataFrame(data=predictions)
Get_data.exportFile(predictions, "test_predictions.csv")

path = path.join(root_path, 'pwsvogelmodel')
model = models.load_model(path)
test_loss = model.evaluate(x=test_weather_dataframe, y=test_bird_dataframe)
