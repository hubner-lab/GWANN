import tensorflow as tf
from tensorflow.keras import layers, models, Input

# Placeholder dimensions
H, W = 28, 28  # Example dimensions for matrix input
samples = 64   # Example size for vector input

# Input layers
matrix_input = Input(shape=(H, W, 1), name="matrix_input")  # (H, W, 1)
vector_input = Input(shape=(samples,), name="vector_input")  # flat vector

# CNN Branch (equivalent to self.sequential)
x1 = layers.Conv2D(3, kernel_size=(2, 2), activation='relu')(matrix_input)
x1 = layers.MaxPooling2D(pool_size=(2, 2))(x1)
x1 = layers.Conv2D(5, kernel_size=(2, 2), activation='relu')(x1)
x1 = layers.MaxPooling2D(pool_size=(2, 2))(x1)
x1 = layers.Conv2D(10, kernel_size=(2, 2), activation='relu')(x1)

x1 = layers.Flatten()(x1)  # Flatten to shape (batch_size, final)
final_dim = x1.shape[-1]   # Final feature size after CNN


# Vector branch (equivalent to self.pop_seq)
x2 = layers.Dense(samples // 2, activation='relu')(vector_input)
x2 = layers.Dropout(0.2)(x2)
x2 = layers.Dense(final_dim, activation='sigmoid')(x2)  # Project to match CNN output size

# Element-wise multiplication (like x1 * x2)
x3 = layers.Multiply()([tf.reshape(x1, (-1, final_dim)), x2])

# Linear head (equivalent to self.lin_seq)
x3 = layers.Dense(10, activation='relu')(x3)
x3 = layers.Dropout(0.2)(x3)
x3 = layers.Dense(1, activation=None)(x3)


# Final binary classification layer
x = layers.Dense(1, activation='sigmoid')(3)

# Build model
model = models.Model(inputs=[matrix_input, vector_input], outputs=x)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()