import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, ReLU, Add, Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from datetime import datetime

# Load dataset
folder = "./ce_challenge/dataset_train_val/"
filename = "dataset_train_val_challenge_5.npz"
data = np.load(folder + filename)

trainData_train = data["trainData_train"]
trainLabels_train = data["trainLabels_train"]
trainData_validation = data["trainData_validation"]
trainLabels_validation = data["trainLabels_validation"]


# Function to create transfer learning model
def create_transfer_model(base_model, input_shape=(612, 14, 2), trainable_layers=5):
    base_model = base_model(input_shape=input_shape, include_top=False, weights=None)
    for layer in base_model.layers[:-trainable_layers]:  # Freeze all but the last trainable_layers
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(trainLabels_train.shape[-1], activation='linear')(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model


# Choose base model (example: EDSR)
from tensorflow.keras.applications import MobileNetV2

base_model_fn = MobileNetV2  # Change this to any other model function

# Create transfer model
transfer_model = create_transfer_model(base_model_fn)
transfer_model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])

# Callbacks
model_type = 'transfer_learning'
timenow = datetime.now().strftime("%H%M%S%d%m%y")
model_filename = f'{timenow}_{model_type}_best_model.keras'

early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(model_filename, monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=40, min_lr=0.00001)

# Train the model
transfer_model.fit(trainData_train, trainLabels_train,
                   epochs=10,
                   batch_size=64,
                   validation_data=(trainData_validation, trainLabels_validation),
                   callbacks=[early_stopping, model_checkpoint, reduce_lr])

# Load best model
model = load_model(model_filename)

# Evaluate model
loss, mae = model.evaluate(trainData_validation, trainLabels_validation)
print(f"Validation Loss: {loss:.4f}, MAE: {mae:.4f}")