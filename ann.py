import os
import pandas as pd
import tensorflow as tf
from helpers import Data_processing
#from helpers import Create_model


# main functions
def get_data(path_to_csv):
    raw_dataframe = pd.read_csv(path_to_csv)
    raw_weather_dataframe = Data_processing.process_time_data(raw_dataframe)
    clean_dataframe = Data_processing.process_weather_data(
        raw_weather_dataframe)
    return clean_dataframe


def build_model():
    model = Create_model
    return model


def train_model(model, train_data):
    # add stuff
    return model


def predict_with_model(model, data):
    # add stuff
    return predicted_data
    pass


path_to_csv = os.path.join(os.getcwd(), "data/weather_data.csv")
weather_data = get_data(path_to_csv)
model = build_model()
trained_model = train_model(model, train_data)
