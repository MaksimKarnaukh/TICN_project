import scipy as scipy
import matplotlib.pyplot as plt
import numpy as np

import h5py
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, ReLU, GlobalAveragePooling2D, Reshape, Dense, multiply, Add, Concatenate, Lambda
from tensorflow.keras.utils import plot_model

import helper
from helper import exec_time_and_y_pred
#%%
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)
#%%
folder = "../dataset_train_val/"
filename = "dataset_train_val_challenge_5.npz"
fullpath = folder+filename
#%%
data = np.load(fullpath)
#%%
# Access arrays by their keys
trainData_train = data["trainData_train"]
trainLabels_train = data["trainLabels_train"]
trainPractical_train = data["trainPractical_train"]
trainLinearInterpolation_train = data["trainLinearInterpolation_train"]
otherLabels_train = data["otherLabels_train"]

trainData_validation = data["trainData_validation"]
trainLabels_validation = data["trainLabels_validation"]
trainPractical_validation = data["trainPractical_validation"]
trainLinearInterpolation_validation = data["trainLinearInterpolation_validation"]
otherLabels_validation = data["otherLabels_validation"]
#%%
# Print shapes to verify
print("Training Data shape:", trainData_train.shape)
print("Validation Data shape:", trainData_validation.shape)
#%%
def SRCNN(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (9, 9), padding='same', activation='relu')(inputs)
    x = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    outputs = Conv2D(2, (5, 5), padding='same', activation='linear')(x)
    model = Model(inputs, outputs, name="SRCNN")
    return model

def DnCNN(input_shape, depth=17, num_filters=64):
    inputs = Input(shape=input_shape)
    x = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(inputs)

    for _ in range(1, depth-1):
        x = Conv2D(num_filters, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

    outputs = Conv2D(input_shape[-1], (3, 3), padding='same')(x)

    # Residual Learning: output the noise, then subtract it from input to get denoised image
    outputs = Add()([inputs, outputs])

    model = Model(inputs, outputs, name="DnCNN")
    return model
#%%
input_shape = (612, 14, 2)  # Example input shape

# Step 1: Train the SRCNN Model
srcnn_model = SRCNN(input_shape)
srcnn_model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
srcnn_model.summary()
#%%
#model name
model_type = 'srcnn_1024'
timenow = datetime.now().strftime("%H%M%S%d%m%y")
model_filename_srcnn = f'{timenow}_{model_type}_best_classifier_model.keras'
#%%
# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
classifier_checkpoint = ModelCheckpoint(model_filename_srcnn, monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=40, min_lr=0.00001)
#%%
## Train the SRCNN part
srcnn_model.fit(trainData_train, trainLabels_train,
          epochs=2,
          batch_size=64,
          shuffle=True,
          validation_data=(trainData_validation, trainLabels_validation),
          callbacks=[early_stopping, classifier_checkpoint, reduce_lr])
#%%
srcnn_model = load_model(model_filename_srcnn)
#%%
# Step 2: Freeze the SRCNN Model and Train the DnCNN Model
for layer in srcnn_model.layers:
    layer.trainable = False

inputs = Input(shape=input_shape)
sr_output = srcnn_model(inputs)

dncnn_model = DnCNN(sr_output.shape[1:])
final_output = dncnn_model(sr_output)

channelnet_model = Model(inputs, final_output, name="ChannelNet")
channelnet_model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
channelnet_model.summary()
#%%
#model name
model_type = 'channelnet_1024'
timenow = datetime.now().strftime("%H%M%S%d%m%y")
model_filename_channelnet = f'{timenow}_{model_type}_best_classifier_model.keras'
#%%
# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
classifier_checkpoint = ModelCheckpoint(model_filename_channelnet, monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=25, min_lr=0.00001)
#%%
###### Train ChannelNet (SRCNN + DnCNN)
channelnet_model.fit(trainData_train, trainLabels_train,
          epochs=2,
          batch_size=64,
          shuffle=True,
          validation_data=(trainData_validation, trainLabels_validation),
          callbacks=[early_stopping, classifier_checkpoint, reduce_lr])
#%%
model = load_model(model_filename_channelnet)
#%%
#Calculate execution time
total_time_se, time_per_sample_se, y_pred = exec_time_and_y_pred(model, trainData_validation)
#%%
print(f"Total execution time: {total_time_se:.6f} seconds")
print(f"Execution time per sample: {time_per_sample_se:.6f} seconds")
#%%
print("CNN")
first_dimension = trainLabels_validation.shape[0]
cnn_mse = mean_squared_error(trainLabels_validation.reshape(first_dimension,-1), y_pred.reshape(first_dimension,-1))
cnn_mae = mean_absolute_error(trainLabels_validation.reshape(first_dimension,-1), y_pred.reshape(first_dimension,-1))
cnn_r2 = r2_score(trainLabels_validation.reshape(first_dimension,-1), y_pred.reshape(first_dimension,-1))
print("MSE: %.4f" % cnn_mse)
print("MAE: %.4f" % cnn_mae)
print('R²: %.4f' % cnn_r2)
#%%
print("Practical CE")
practical_mse =  mean_squared_error(trainLabels_validation.reshape(first_dimension,-1), trainPractical_validation.reshape(first_dimension,-1))
practical_mae = mean_absolute_error(trainLabels_validation.reshape(first_dimension,-1), trainPractical_validation.reshape(first_dimension,-1))
practical_r2 =  r2_score(trainLabels_validation.reshape(first_dimension,-1), trainPractical_validation.reshape(first_dimension,-1))

print("MSE: %.4f" % practical_mse)
print("MAE: %.4f" % practical_mae)
print('R²: %.4f' % practical_r2)
#%%
print("Linear CE")
mse_linear = mean_squared_error(trainLabels_validation.reshape(first_dimension,-1), trainLinearInterpolation_validation.reshape(first_dimension,-1))
mae_linear = mean_absolute_error(trainLabels_validation.reshape(first_dimension,-1), trainLinearInterpolation_validation.reshape(first_dimension,-1))
r2_linear = r2_score(trainLabels_validation.reshape(first_dimension,-1), trainLinearInterpolation_validation.reshape(first_dimension,-1))
print("MSE: %.4f" % mse_linear)
print("MAE: %.4f" % mae_linear)
print('R²: %.4f' % r2_linear)
#%%
print("Comparing and improvement")
print(f'CNN MSE: {cnn_mse}, Practical: {practical_mse}, Linear: {mse_linear}')
print(f'CNN vs. Linear MSE {(mse_linear-cnn_mse)*100/mse_linear}, CNN vs. Practical {(practical_mse-cnn_mse)*100/practical_mse}')
#%%
# Generate a random index
#random_index = np.random.randint(0, trainData_validation.shape[0])
random_index = 8
input_sample = trainData_validation[random_index]
label_sample = trainLabels_validation[random_index]
practical_sample = trainPractical_validation[random_index]
linearInterpol_sample = trainLinearInterpolation_validation[random_index]
predicted_sample = model.predict(np.expand_dims(input_sample, axis=0))[0]
#%%
# Compute the magnitude using the absolute value of the last dimension
input_magnitude = np.linalg.norm(input_sample, axis=-1)
label_magnitude = np.linalg.norm(label_sample, axis=-1)
practical_magnitude = np.linalg.norm(practical_sample, axis=-1)
linearInterpol_magnitude = np.linalg.norm(linearInterpol_sample, axis=-1)
predicted_magnitude = np.linalg.norm(predicted_sample, axis=-1)
#%%
cmax = np.max(np.concatenate((np.abs(label_magnitude), np.abs(practical_magnitude), np.abs(predicted_magnitude)), axis=0))
cmin = np.min(np.concatenate((np.abs(label_magnitude), np.abs(practical_magnitude), np.abs(linearInterpol_magnitude), np.abs(predicted_magnitude)), axis=0))
#%%
# Calculate the Mean Squared Error
mse_linear = np.mean((label_sample - linearInterpol_sample) ** 2)
print(f'MSE Linear: {mse_linear}')

mse_prac = np.mean((label_sample - practical_sample) ** 2)
print(f'MSE CE practical: {mse_prac}')

# Calculate the Mean Squared Error
mse_nn = np.mean((label_sample - predicted_sample) ** 2)
print(f'MSE DNN: {mse_nn}')
#%%
# Plotting
#cmap = parula_colormap
cmap = "jet"
font_size = 18
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(15,6), sharey=True)

axs[0].imshow(np.abs(linearInterpol_magnitude), cmap=cmap, vmin=cmin, vmax=cmax)
axs[0].set_aspect(0.05)
axs[0].set_title(f'Linear CE\nMSE: {mse_linear:.4f}',fontsize=font_size)

axs[1].imshow(np.abs(practical_magnitude), cmap=cmap, vmin=cmin, vmax=cmax)
axs[1].set_aspect(0.05)
axs[1].set_title(f'Practical CE\nMSE: {mse_prac:.4f}',fontsize=font_size)

axs[2].imshow(np.abs(predicted_magnitude), cmap=cmap, vmin=cmin, vmax=cmax)
axs[2].set_aspect(0.05)
axs[2].set_title(f'CNN predicted\nMSE: {mse_nn:.4f}',fontsize=font_size)

axs[3].imshow(np.abs(label_magnitude), cmap=cmap, vmin=cmin, vmax=cmax)
axs[3].set_aspect(0.05)
axs[3].set_title('Actual Channel',fontsize=font_size)

plt.show()
#%%
# Unique SNR values
unique_snr_values, unique_snr_counts = np.unique(otherLabels_validation[:, 1], return_counts=True)
#%%
# Calculate MSE for each SNR value
mse_results = {
    "SNR": [],
    "MSE_y_pred": [],
    "MSE_practical": [],
    "MSE_linear_interpolation": []
}

for snr, idx in zip(unique_snr_values, unique_snr_counts):
    indices = otherLabels_validation[:, 1] == snr

    # Determine the first dimension for reshaping
    first_dimension = idx

    mse_y_pred = mean_squared_error(
        trainLabels_validation[indices].reshape(first_dimension, -1),
        y_pred[indices].reshape(first_dimension, -1)
    )
    mse_practical = mean_squared_error(
        trainLabels_validation[indices].reshape(first_dimension, -1),
        trainPractical_validation[indices].reshape(first_dimension, -1)
    )
    mse_linear_interpolation = mean_squared_error(
        trainLabels_validation[indices].reshape(first_dimension, -1),
        trainLinearInterpolation_validation[indices].reshape(first_dimension, -1)
    )

    mse_results["SNR"].append(snr)
    mse_results["MSE_y_pred"].append(mse_y_pred)
    mse_results["MSE_practical"].append(mse_practical)
    mse_results["MSE_linear_interpolation"].append(mse_linear_interpolation)

# Plotting the results with log10 scale for y-axis
plt.figure(figsize=(10, 6))
plt.plot(mse_results["SNR"], mse_results["MSE_y_pred"], label="MSE y_pred", marker='o')
plt.plot(mse_results["SNR"], mse_results["MSE_practical"], label="MSE Practical", marker='x')
plt.plot(mse_results["SNR"], mse_results["MSE_linear_interpolation"], label="MSE Linear Interpolation", marker='s')
plt.xlabel("SNR values")
plt.ylabel("MSE (log10 scale)")
plt.yscale('log')
plt.title("MSE vs SNR values")
plt.legend()
plt.grid(True)
plt.show()