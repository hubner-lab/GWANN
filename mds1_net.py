# model_builder.py
from keras import layers, Input, Model
class ModelBuilder:
    def __init__(self, height, width, pop_vector_length, samples):
        self.height = height
        self.width = width
        self.pop_vector_length = pop_vector_length
        self.samples = samples

    def build(self):
        
        matrix_input = Input(shape=(self.height, self.width, 1))
        vector_input = Input(shape=(self.pop_vector_length,))

        x1 = layers.Conv2D(3, kernel_size=(2, 2), activation='relu')(matrix_input)
        x1 = layers.MaxPooling2D(pool_size=(2, 2))(x1)
        x1 = layers.Conv2D(5, kernel_size=(2, 2), activation='relu')(x1)
        x1 = layers.MaxPooling2D(pool_size=(2, 2))(x1)
        x1 = layers.Conv2D(10, kernel_size=(2, 2), activation='relu')(x1)
        x1 = layers.Flatten()(x1)
        final_dim = x1.shape[-1]  


        x2 = layers.Dense(self.samples // 2, activation='relu')(vector_input)
        x2 = layers.Dropout(0.2)(x2)
        x2 = layers.Dense(final_dim, activation='sigmoid')(x2)

        x3 = layers.Multiply()([x1, x2])

        x3 = layers.Dense(10, activation='relu')(x3)
        x3 = layers.Dropout(0.2)(x3)
        

        output = layers.Dense(2, activation='softmax')(x3)

        return Model(inputs=[matrix_input, vector_input], outputs=output)


    def model_summary(self):
        model = self.build()
        model.summary()
        return model
    