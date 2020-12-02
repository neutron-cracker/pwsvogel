import os

from pandas.core.frame import DataFrame
from six import print_
import data
import pandas
from data import Convert_data
from data import Get_data
from model import Model as modelClass
import tensorflow
import numpy
import matplotlib.pyplot as plt
from tensorflow import keras
# main functions

# --------------------------------------------------------------------------------------------------------
# Read CSV file for weather and parse dates into pandas dataframe.
# --------------------------------------------------------------------------------------------------------

root_path = os.getcwd()
path_to_csv_weather = os.path.join(root_path, "data/weather_data.csv")
path_to_csv_bird = os.path.join(
    root_path, "data/bird_migration_per_specie.csv")

weather_data_raw = data.Get_data.raw_weather_data(path_to_csv_weather)
bird_data_unfiltered = data.Get_data.bird_data(path_to_csv_bird)


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
# train_dataset = Convert_data.dataframe_to_tf_dataset(
#     train_weather_dataframe, train_bird_dataframe)  # , train_dataframe)
epochs = 200

# export dataframe


Get_data.exportFile(train_bird_dataframe, "bird.csv")
Get_data.exportFile(train_weather_dataframe, "weather.csv")


# --------------------------------------------------------------------------------------------------------
# Train model and save the model.
# --------------------------------------------------------------------------------------------------------

print(len(train_bird_dataframe.columns))
print(train_bird_dataframe.columns[38])
for column in range((len(train_bird_dataframe.columns) - 1), 10, -1):
    train_bird_dataframe = train_bird_dataframe.drop(
        train_bird_dataframe.columns[column], axis=1)
for column in range((len(validation_bird_dataframe.columns) - 1), 10, -1):
    validation_bird_dataframe = validation_bird_dataframe.drop(
        validation_bird_dataframe.columns[column], axis=1)
for column in range((len(test_bird_dataframe.columns) - 1), 10, -1):
    test_bird_dataframe = test_bird_dataframe.drop(
        test_bird_dataframe.columns[column], axis=1)


print(validation_bird_dataframe.shape)

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

model.save(os.path.join(root_path, "pwsvogelmodel"))
predictions = model.predict(x=test_weather_dataframe)
predictions = pandas.DataFrame(data=predictions)
Get_data.exportFile(predictions, "test_predictions.csv")

path = os.path.join(os.getcwd(), 'pwsvogelmodel')
model = keras.models.load_model(path)
validation_loss = model.evaluate(x=test_weather_dataframe, y=test_bird_dataframe)
print(validation_loss)
