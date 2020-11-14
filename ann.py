# import os
# import datetime

import pandas as pd
# import numpy as np
# import seaborn as sns
import tensorflow as tf 
# import matplotlib as mpl 
# import matplotlib.pyplot as plt 
# import IPython
# import IPython.display


class Data_processing:
    
    days_in_year = 365.2425 # use for date convertion
    
    def data_cleaning (csv_raw_data):
        raw_data = pd.read_csv(csv_raw_data)
        # remove extremes
        # remove wind rotation under 90
    
    def process_to_requested_format(csv_raw_data): #requested input: collums: date (YYYYMMDD), wind_veloctity (m/s), wind_rotation(degree), temperature(degree_C), rainfall(mm)
        cleaned_data = Data_processing.data_cleaning(csv_raw_data) # TODO #1
        # make sin/cos from data and weather

class Create_model:
    def __init__(self):
        self.model = ()

    model = tf.keras.Sequential()
    # the layers of the model

# add class for requesting weather-data from knmi

# main functions
def build_model():
    csv_raw_data = path_to_csv # create a fix
    clean_data = Data_processing.process_to_requested_format(csv_raw_data)
    model = Create_model()

def train_model():
    # add stuff
    pass


def predict_with_model():
    # add stuff
    pass
