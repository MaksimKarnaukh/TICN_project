import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, ReLU, Add, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import KLDivergence, MeanSquaredError

import h5py

# Load dataset (assumes same structure as previous code)
folder = "../ce_challenge/dataset_train_val/"
filename = "dataset_train_val_challenge_5.npz"
data = np.load(folder + filename)
trainData_train = data["trainData_train"]
trainLabels_train = data["trainLabels_train"]
trainData_validation = data["trainData_validation"]
trainLabels_validation = data["trainLabels_validation"]

teacher_model = load_model("../ce_challenge/notebooks/182518080325_channelnet_1024_best_classifier_model.keras")

# model_path = "../ce_challenge/trained_models/ChannelNet_215.keras"
# with h5py.File(model_path, 'r') as f:
#     print(f.keys())  # Check if 'model_weights' exists

# Define a small student model
def StudentModel(input_shape=(612, 14, 2)):
    inputs = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(2, (3, 3), padding='same')(x)  # Match output channels
    model = Model(inputs, x, name="StudentModel")
    return model

student_model = StudentModel()

# Distillation loss function
def distillation_loss(y_true, y_pred, teacher_pred, alpha=0.5, temperature=3.0):
    print(y_true.shape, y_pred.shape, teacher_pred.shape)

    batch_size = tf.shape(y_pred)[0]  # Get current batch size
    batch_size = tf.cast(batch_size, tf.int32)
    teacher_batch = teacher_pred[:batch_size]  # Match batch size dynamically

    # kl_loss = KLDivergence()(tf.nn.softmax(teacher_batch / temperature), tf.nn.softmax(y_pred / temperature))
    # mse_loss = MeanSquaredError()(y_true, y_pred)
    # return alpha * kl_loss + (1 - alpha) * mse_loss

    # Apply temperature scaling
    y_pred_soft = tf.nn.softmax(y_pred / temperature, axis=-1)
    teacher_soft = tf.nn.softmax(teacher_batch / temperature, axis=-1)

    return tf.keras.losses.kl_divergence(teacher_soft, y_pred_soft)

# Generate teacher predictions (soft labels)
teacher_predictions = teacher_model.predict(trainData_train, batch_size=64)

# Compile student model
optimizer = Adam()
student_model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: distillation_loss(y_true, y_pred, teacher_predictions))

# Train student model
student_model.fit(trainData_train, trainLabels_train,
                  epochs=10, batch_size=64, validation_data=(trainData_validation, trainLabels_validation))

# Save student model
student_model.save("student_model_distilled.keras")