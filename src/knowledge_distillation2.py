import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, ReLU, Add, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import KLDivergence, MeanSquaredError

import numpy as np
import random
import keras

print(keras.__version__)

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

tf.keras.utils.set_random_seed(42)

# Load dataset (assumes same structure as previous code)
folder = "../ce_challenge/dataset_train_val/"
filename = "dataset_train_val_challenge_5.npz"
data = np.load(folder + filename)
trainData_train = data["trainData_train"]
trainLabels_train = data["trainLabels_train"]
trainData_validation = data["trainData_validation"]
trainLabels_validation = data["trainLabels_validation"]

teacher_model = load_model("../ce_challenge/notebooks/182518080325_channelnet_1024_best_classifier_model.keras")

def StudentModel(input_shape=(612, 14, 2)):
    inputs = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(2, (3, 3), padding='same')(x)  # Match output channels
    model = Model(inputs, x, name="StudentModel")
    return model

student_model = StudentModel()

# # randomly predefine weights for the student model
# student_model.set_weights([np.random.rand(*w.shape) for w in student_model.get_weights()])

# Generate soft labels using teacher model
X_train, y_train = trainData_train, trainLabels_train
soft_labels = teacher_model.predict(X_train, batch_size=64)

# Use soft labels to train the student model
student_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
student_model.fit(X_train, soft_labels, epochs=10, batch_size=64, shuffle=False)

# Evaluate the student model
X_val, y_val = trainData_validation, trainLabels_validation
teacher_model.evaluate(X_val, y_val)
student_model.evaluate(X_val, y_val)