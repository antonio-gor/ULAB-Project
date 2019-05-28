################
## Packages to be used
####

from keras.models import load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

################
## Loading array data
####

## Function to print 
def print_shapes(train, val, test):
    print('Train shape: {}'.format(train.shape))
    print('Val shape  : {}'.format(val.shape))
    print('Test shape : {}'.format(test.shape))

## Load the flux data and labels 
train_flux_data = np.load("npy_arrays/train_data.npy")
train_label = np.load("npy_arrays/train_labels.npy")
val_flux_data = np.load("npy_arrays/val_data.npy")
val_label = np.load("npy_arrays/val_labels.npy")
test_flux_data = np.load("npy_arrays/test_data.npy")
test_label = np.load("npy_arrays/test_labels.npy")
print('Original Arrays')
print_shapes(train_flux_data, val_flux_data, test_flux_data)

## Reshaping for the model
print('\nReshaped Arrays')
train_flux_data = train_flux_data.reshape(train_flux_data.shape[0], train_flux_data.shape[1], 1)
val_flux_data = val_flux_data.reshape(val_flux_data.shape[0], val_flux_data.shape[1], 1)
test_flux_data = test_flux_data.reshape(test_flux_data.shape[0], test_flux_data.shape[1], 1)
print_shapes(train_flux_data, test_flux_data, val_flux_data)

################
## Defining the model
####

#activation=tf.nn.relu
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(2001, 1)),
    tf.keras.layers.Conv1D(filters=16, kernel_size=5, activation=tf.nn.relu),
    tf.keras.layers.Conv1D(filters=16, kernel_size=5, activation=tf.nn.relu),
    tf.keras.layers.MaxPooling1D(pool_size=5, strides=2),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation=tf.nn.relu),
    tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation=tf.nn.relu),
    tf.keras.layers.MaxPooling1D(pool_size=5, strides=2),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation=tf.nn.relu),
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation=tf.nn.relu),
    tf.keras.layers.MaxPooling1D(pool_size=5, strides=2),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation=tf.nn.relu),
    tf.keras.layers.Conv1D(filters=128, kernel_size=5, activation=tf.nn.relu),
    tf.keras.layers.MaxPooling1D(pool_size=5, strides=2),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
])


################
## Compiling the model
####

## Compiles, displays, fits, and saves the model
def fit_model(data, labels, batch_size=8, epochs=5):

    ## Compiling the model
    model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.0001, epsilon=1e-08),
                  loss='mse',
                  metrics=['accuracy'])

    ## Displaying the model's summary
    print(model.summary())

    ## Fit the model and save the output
    history = model.fit(data, 
                        labels, 
                        validation_data=(val_flux_data, val_label), 
                        batch_size=batch_size, 
                        epochs=epochs)

    ## Saving the model as an HDF5 file
    model.save('model.h5')

    ## Return model history object
    return history
