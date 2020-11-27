import tensorflow
from tensorflow import keras
from tensorflow.keras import layers


class Model:  # base class for model
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
    def train_model(model, input_dataset, epochs):
        model.fit(x=input_dataset, epochs=epochs, verbose=2)
        return model
	# --------------------------------------------------------------------------------------------------------
	# Predict with model.
	# --------------------------------------------------------------------------------------------------------
    def predict_with_model(model, input_data):
        model.predict(x=input_data)
        return predicted_data
