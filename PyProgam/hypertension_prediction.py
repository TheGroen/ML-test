import pandas as pd
import tensorflow as tf

# Import data
data = pd.read_csv('F:patient_data.csv')

# Identifying the target values.
target = data['Hyper']

# Dimensions of dataset
n = data.shape[0]
p = data.shape[1]

# Removing the Name column as it isn't important.
data = data.drop('Name', 1)

# Removing the Hyper column as it is the target.
data = data.drop('Hyper', 1)

# Converting the dataset to an array.
data = data.values
target = target.values

# Creating the the model.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

# Compile the model.
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

# Run the model fitting.
model.fit(data, target, epochs=300)

# Input data for which a prediction needs to be made.
guess = [[69, 178, 1, 0, 1, 59]]

# Print the input values
print('\nPatient input data:', guess)

# Make a prediction from the model.
chance = model.predict(guess)

x = chance[0][1] * 100

# Print out the results of the prediction.
print('\nThe patient percentage chance of being hypertensive is: %.2f ' % x)
