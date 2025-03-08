import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Multiply, GlobalAveragePooling2D, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np

def ChannelAttention(input_tensor, reduction_ratio=8):
    channels = input_tensor.shape[-1]
    x = GlobalAveragePooling2D()(input_tensor)
    x = Dense(channels // reduction_ratio, activation='relu')(x)
    x = Dense(channels, activation='sigmoid')(x)
    x = Reshape((1, 1, channels))(x)
    return Multiply()([input_tensor, x])

def AttentionDnCNN(input_shape, depth=10, num_filters=48):
    inputs = Input(shape=input_shape)
    x = Conv2D(num_filters, (3, 3), padding='same', activation='relu')(inputs)
    
    for _ in range(1, depth - 1):
        x = Conv2D(num_filters, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = ChannelAttention(x)  # Lightweight Attention Mechanism
    
    outputs = Conv2D(input_shape[-1], (3, 3), padding='same')(x)
    outputs = Add()([inputs, outputs])  # Residual Learning
    
    model = Model(inputs, outputs, name="AttentionDnCNN")
    return model

# Example Usage

# Load dataset
data = np.load("../ce_challenge/dataset_train_val/dataset_train_val_challenge_5.npz")

print("Keys in dataset:", list(data.keys()))

X_train, Y_train = data["trainData_train"], data["trainLabels_train"]
X_val, Y_val = data["trainData_validation"], data["trainLabels_validation"]

print(f"Training Data Shape: {X_train.shape}, Labels Shape: {Y_train.shape}")
print(f"Validation Data Shape: {X_val.shape}, Labels Shape: {Y_val.shape}")

# Create and compile model
input_shape = X_train.shape[1:]
model = AttentionDnCNN(input_shape)
model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError(), metrics=["mse"])

# Train the model
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=32)

test_loss, test_mse = model.evaluate(X_val, Y_val)
print(f"Test MSE: {test_mse}")