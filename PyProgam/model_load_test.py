import tensorflow as tf
import pandas as pd

# Load the test dataset.
data = pd.read_csv('F:test_data.csv')

# Create the target value set.
target = data['Hyper']

# Removing the Name column as it isn't important.
data = data.drop('Name', 1)

# Removing the Hyper column as it is the target.
data = data.drop('Hyper', 1)

# Converting the dataset to an array.
data = data.values
target = target.values

# Load the model.
new_model = tf.keras.models.load_model('F:saved_models/my_model')

# Check the model's architecture.
new_model.summary()

# Evaluate the loaded model
loss, acc = new_model.evaluate(data, target, verbose=2)

x = acc * 100

# Prints the model accuracy.
print('Restored model, accuracy: %.2f' % x)
