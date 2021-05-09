import tensorflow as tf

# Load the model.
new_model = tf.keras.models.load_model('F:saved_models/my_model')

# Check the model's architecture.
new_model.summary()

# Input data for which a prediction needs to be made.
guess = [[69, 154, 1, 1, 1, 61]]

# Use the input data to make a prediction.
chance = new_model.predict(guess)

x = chance[0][1] * 100

# Print out the results of the prediction.
print('\nThe patient percentage chance of being hypertensive is: %.2f' % x, '%')
