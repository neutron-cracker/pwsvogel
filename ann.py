# import os
# import datetime
import pandas as pd
# import numpy as np
# import seaborn as sns
import tensorflow as tf 
import helpers
from helpers import Data_processing
from helpers import Create_model
# import matplotlib as mpl 
# import matplotlib.pyplot as plt 
# import IPython
# import IPython.display

# main functions
def build_model(path_to_csv):
    csv_raw_data = path_to_csv
    clean_data = Data_processing.process_to_requested_format(csv_raw_data)
    model = Create_model()

def train_model():
    # add stuff
    pass


def predict_with_model():
    # add stuff
    pass
