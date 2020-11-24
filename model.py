import tensorflow
from tensorflow import keras
from tensorflow.keras import layers


class Model:  # base class for model

    # def __init__(self):
    #    self.model = self.build_model()

# --------------------------------------------------------------------------------------------------------
# Build the model
# --------------------------------------------------------------------------------------------------------

    def build_model():
        model = tensorflow.keras.Sequential([
            keras.layers.Flatten(input_shape=(30,)),
            keras.layers.Dense(64, activation='sigmoid'),
            keras.layers.Dense(64, activation='sigmoid'),
            keras.layers.Dense(66, activation='sigmoid')
        ])
        print(model.summary())  # can eventually be removed, or in logger https://docs.python.org/3/library/logging.html
        model.compile(optimizer='adam', loss=tensorflow.keras.losses.BinaryCrossentropy(
            from_logits=True), metrics=['accuracy'])
        return model

# --------------------------------------------------------------------------------------------------------
# Train the model.
# --------------------------------------------------------------------------------------------------------

    def train_model(model, input_train_data, target_data, epochs, validation_data):
        model.fit(x=input_train_data, y=target_data, epochs=epochs, verbose=2, validation_data=validation_data)
        return model

# --------------------------------------------------------------------------------------------------------
# Predict with model.
# --------------------------------------------------------------------------------------------------------

    def predict_with_model(model, input_data):
        model.predict(x=input_data)
        return predicted_data

