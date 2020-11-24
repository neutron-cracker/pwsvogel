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
        model.compile() #optimizer='Nadam', loss=tensorflow.keras.losses.BinaryCrossentropy(
            #from_logits=True), metrics=['accuracy']
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

# -------------------------------------------
# DELETE
# -------------------------------------------
def _filter_grads(grads_and_vars):
  """Filter out iterable with grad equal to None."""
  grads_and_vars = tuple(grads_and_vars)
  if not grads_and_vars:
    return grads_and_vars
  filtered = []
  vars_with_empty_grads = []
  for grad, var in grads_and_vars:
    if grad is None:
      vars_with_empty_grads.append(var)
    else:
      filtered.append((grad, var))
  filtered = tuple(filtered)
  if not filtered:
    raise ValueError("No gradients provided for any variable: %s." %
                     ([v.name for _, v in grads_and_vars],))
  if vars_with_empty_grads:
    logging.warning(
        ("Gradients do not exist for variables %s when minimizing the loss."),
        ([v.name for v in vars_with_empty_grads]))
  return filtered