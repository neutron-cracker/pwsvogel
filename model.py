import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
import numpy


class Model:  # base class for model

    # def __init__(self):
    #    self.model = self.build_model()

    # --------------------------------------------------------------------------------------------------------
    # Build the model
    # --------------------------------------------------------------------------------------------------------

    def build_model():
        model = tensorflow.keras.Sequential([
            keras.layers.Flatten(input_shape=(31,)),
            keras.layers.Dense(64, activation='sigmoid'),
            keras.layers.Dense(64, activation='sigmoid'),
            keras.layers.Dense(66, activation='sigmoid')
        ])
        # can eventually be removed, or in logger https://docs.python.org/3/library/logging.html
        print(model.summary())
        model.compile()  # optimizer='Nadam', loss=tensorflow.keras.losses.BinaryCrossentropy(
        # from_logits=True), metrics=['accuracy']
        return model

# --------------------------------------------------------------------------------------------------------
# Train the model.
# --------------------------------------------------------------------------------------------------------

    def train_model(model, input_train_data, target_data, epochs, validation_data):
        # validation_data=validation_data)
        model.fit(x=input_train_data, epochs=epochs, verbose=2)
        return model

        # https://stackoverflow.com/questions/36921951/truth-value-of-a-series-is-ambiguous-use-a-empty-a-bool-a-item-a-any-o

# --------------------------------------------------------------------------------------------------------
# Predict with model.
# --------------------------------------------------------------------------------------------------------

    def predict_with_model(model, input_data):
        model.predict(x=input_data)
        return predicted_data
