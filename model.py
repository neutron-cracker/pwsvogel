import tensorflow
from tensorflow import keras
from tensorflow.keras import layers


class Model:  # base class for model
    # --------------------------------------------------------------------------------------------------------
    # Build the model
    # --------------------------------------------------------------------------------------------------------
    def build_model():
        model = tensorflow.keras.Sequential([
            layers.Flatten(input_shape=(30,)),
            layers.Dense(1024, activation='sigmoid'),
            layers.Dense(1024, activation='sigmoid'),
            layers.Dense(1024, activation='sigmoid'),
            layers.Dense(66)
        ])

        model.compile(optimizer='adadelta', loss=keras.losses.MeanSquaredLogarithmicError(  # adadelta, adagrad
        ), metrics=['accuracy'])
        return model

        # --------------------------------------------------------------------------------------------------------
        # Predict with model.
        # --------------------------------------------------------------------------------------------------------
    def predict_with_model(model, input_data):
        predicted_data = model.predict(x=input_data)
        return predicted_data
