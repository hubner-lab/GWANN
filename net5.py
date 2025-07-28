from keras import layers, Input, Model
import tensorflow as tf

class ModelBuilder:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def build(self):
        l2_reg = tf.keras.regularizers.l2(0.01)

        matrix_input = Input(shape=(self.height, self.width, 1))

        # First Conv Block
        x = layers.Conv2D(16, kernel_size=(3, 3), activation='tanh', padding='same',
                          kernel_regularizer=l2_reg, bias_regularizer=l2_reg)(matrix_input)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Second Conv Block
        x = layers.Conv2D(32, kernel_size=(3, 3), activation='tanh', padding='same',
                          kernel_regularizer=l2_reg, bias_regularizer=l2_reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Global pooling
        x = layers.GlobalAveragePooling2D()(x)

        # Fully connected
        x = layers.Dense(64, activation='relu',
                         kernel_regularizer=l2_reg, bias_regularizer=l2_reg,
                         activity_regularizer=tf.keras.regularizers.l1(0.005))(x)
        x = layers.Dropout(0.5)(x)

        # Output
        output = layers.Dense(2, activation='softmax',
                              kernel_regularizer=l2_reg, bias_regularizer=l2_reg)(x)

        return Model(inputs=matrix_input, outputs=output)

    def model_summary(self):
        model = self.build()
        model.summary()
        return model
