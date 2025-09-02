from keras import layers, Input, Model
import tensorflow as tf

class ModelBuilder:


    def __init__(self, height, width):
        self.height = height
        self.width = width


    def build(self):
        matrix_input = Input(shape=(self.height, self.width, 1))

        x = layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same')(matrix_input)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        
        x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.Dropout(rate=0.5)(x)
        
        output = layers.Dense(2, activation='softmax')(x)

        return Model(inputs=matrix_input, outputs=output)


    def model_summary(self):
        model = self.build()
        model.summary()
        return model
