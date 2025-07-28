# model_builder.py
from keras import layers, Input, Model
class ModelBuilder:
    def __init__(self, height, width):
        self.height = height
        self.width = width


    def build(self):
        matrix_input = Input(shape=(self.height, self.width, 1))
        x = layers.Conv2D(3, kernel_size=(2, 2), activation='relu')(matrix_input)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(5, kernel_size=(2, 2), activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(10, kernel_size=(2, 2), activation='relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(10)(x)
        x = layers.Dropout(rate=0.2)(x)
        output = layers.Dense(2, activation='softmax')(x)

        return Model(inputs=matrix_input, outputs=output)


    def model_summary(self):
        model = self.build()
        model.summary()
        return model
    