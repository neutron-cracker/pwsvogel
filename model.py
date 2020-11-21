class Model:  # base class for model

    def __init__(self):
        self.model = build_model()

    def build_model():
        model = tf.keras.Sequential([
            keras.layers.Flatten(input_shape=30),
            keras.layers.Dense(64, activation='sigmoid'),
            keras.layers.Dense(64, activation='sigmoid'),
            keras.layers.Dense(66, activation='sigmoid')
        ])
        return model

    def train_model(model, train_data):
        # add stuff
        return model

    def predict_with_model(model, data):
        # add stuff
        return predicted_data
        pass
