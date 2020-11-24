import os
import data
import pandas as pandas
from data import Data_processing as Data_processing
from model import Model as modelClass
import tensorflow as tensorflow
# main functions


def get_data(path_to_csv):
    raw_dataframe = pandas.read_csv(path_to_csv)
    raw_dataframe.info()
    raw_weather_dataframe = Data_processing.process_time_data(raw_dataframe)
    clean_dataframe = Data_processing.process_weather_data(
        raw_weather_dataframe)
    transformed_data_frame = data.transform_data_frame.transform_data(
        clean_dataframe)
    dataset = Data_processing.convert_pandas_dataframe_to_tf_dataset(
        transformed_data_frame)
    return dataset


path_to_csv = os.path.join(os.getcwd(), "data/weather_data.csv")
weather_data = get_data(path_to_csv)
# type thing to split data
modelClass = modelClass.build_model()
tensorflow.keras.utils.plot_model(
    modelClass, to_file='model.png'
)

# ERROR: Exception has occurred: AttributeError; 'Sequential' object has no attribute 'train_model' TODO: fix this error. I don't know how...
trained_model = train_model(model, train_data)
