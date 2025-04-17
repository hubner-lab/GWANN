# from dataset import Dataset
from dataset3 import Dataset
from sklearn.model_selection import train_test_split 
from tensorflow import keras, reshape
from keras import layers
import numpy as np
from const import MODEL_PATH_TENSOR_DIR
from mylogger import Logger
import os 
from keras.callbacks import ModelCheckpoint
from utilities import json_update, json_get
import tensorflow as tf
class Train:
    def __init__(self, model_name:str, total_simulations:int, CausalSamples: int, columns:int,
                 batch_size:int, epochs:int, simPath: str = "./simulation/data/", test_ratio:float = 0.2):
        self.model_name = model_name
        self.dataset = Dataset(total_simulations, CausalSamples, columns, simPath)
        self.batch_size = batch_size
        self.epochs = epochs
        self.test_ratio = test_ratio

    def run(self):
        Logger(f'Message:', f"{os.environ['LOGGER']}").info("Starting training process...")
        Logger(f'Message:', f"{os.environ['LOGGER']}").info(f"Loading data from {self.dataset.simPath}")
        self.dataset.load_data()
        Logger(f'Message:', f"{os.environ['LOGGER']}").info("Data loaded successfully.") 
        Logger(f'Message:', f"{os.environ['LOGGER']}").info("Splitting data into training and testing sets...")

        # Split the data; note that here we assume all three arrays have compatible sizes.
        X_train, X_test, pop_train, pop_test, y_train, y_test = train_test_split(
            self.dataset.X.numpy(),
            self.dataset.pop.numpy(),
            self.dataset.y.numpy(),
            test_size=self.test_ratio,
            random_state=42,
            stratify=self.dataset.y.numpy()
        )
        pop_train = np.squeeze(pop_train, axis=1)
        pop_test = np.squeeze(pop_test, axis=1)
        samples = json_get("samples")
        Logger(f'Message:', f"{os.environ['LOGGER']}").debug(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")
        Logger(f'Message:', f"{os.environ['LOGGER']}").info("Data split successfully.")

        self.height, self.width = self.dataset.X.shape[1:3]
        Logger(f'Message:', f"{os.environ['LOGGER']}").info(f"Creating model with input shape: {self.height}x{self.width}x1")

        # Image branch input (matrix_input)
        matrix_input = keras.Input(shape=(self.height, self.width, 1))
        # Population vector branch input: change to a 1D vector of length pop_features (e.g. 300)
        vector_input = keras.Input(shape=(self.dataset.pop.shape[2],))

        # Image branch: convolutional processing
        x1 = layers.Conv2D(3, kernel_size=(2, 2), activation='relu')(matrix_input)
        x1 = layers.MaxPooling2D(pool_size=(2, 2))(x1)
        x1 = layers.Conv2D(5, kernel_size=(2, 2), activation='relu')(x1)
        x1 = layers.MaxPooling2D(pool_size=(2, 2))(x1)
        x1 = layers.Conv2D(10, kernel_size=(2, 2), activation='relu')(x1)
        x1 = layers.Flatten()(x1)
        final_dim = x1.shape[-1]  # Should be an integer

        # Vector branch: process the population vector
        x2 = layers.Dense(samples // 2, activation='relu')(vector_input)
        x2 = layers.Dropout(0.2)(x2)
        x2 = layers.Dense(final_dim, activation='sigmoid')(x2)

        # Combine both branches element-wise; both x1 and x2 should be (None, final_dim)
        x3 = layers.Multiply()([x1, x2])

        # Linear head to produce output
        x3 = layers.Dense(10, activation='relu')(x3)
        x3 = layers.Dropout(0.2)(x3)
        
        # Final binary classification layer with sigmoid activation
        x = layers.Dense(1, activation='sigmoid')(x3)

        # Adjust training data: expand image dims to include a channel
        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)
        # Note: pop_train and pop_test already have shape (num_samples, pop_features)

        model = keras.Model(inputs=[matrix_input, vector_input], outputs=x)
        model.summary()
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

        checkpoint_path = f"{MODEL_PATH_TENSOR_DIR}/{self.model_name}.h5"  
        checkpoint = ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_accuracy',           # or 'val_accuracy' / 'val_acc'
                save_best_only=True,
                save_weights_only=False,
                verbose=1
                )

        train_dataset = tf.data.Dataset.from_tensor_slices(((X_train, pop_train), y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices(((X_test, pop_test), y_test))
        val_dataset = val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)


        Logger(f'Message:', f"{os.environ['LOGGER']}").info("Training model...")
        # model.fit([X_train, pop_train],y_train, 
        #           validation_data=([X_test, pop_test], y_test),
        #           epochs=self.epochs,
        #           batch_size=self.batch_size,
        #           callbacks=[checkpoint],
        #           verbose=1,
        #           use_multiprocessing=True,
        #           workers=4,
        #           )

        cpus = os.cpu_count() or 4

        model.fit(
                    x=[X_train, pop_train],
                    y=y_train,
                    validation_data=([X_test, pop_test], y_test),
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    callbacks=[checkpoint],
                    verbose=1,
                    workers=cpus,
                    use_multiprocessing=True,
                )



        json_update("model_name", f"{MODEL_PATH_TENSOR_DIR}/{self.model_name}.h5")
        Logger(f'Message:', f"Model name updated to {MODEL_PATH_TENSOR_DIR}/{self.model_name}.h5")
        model.save(f"{MODEL_PATH_TENSOR_DIR}/{self.model_name}.h5")
        
        test_loss, test_acc = model.evaluate([X_test, pop_test], y_test)
        print('test loss', test_loss)
        print('test acc', test_acc)
        Logger(f'Message:', f"{os.environ['LOGGER']}").info("Model training completed successfully.")

if __name__ == '__main__':
    total_simulations = 1000
    samples = 8
    columns = 20
    batch_size = 128
    epochs = 100
    test_size = 0.2

    trainer = Train(total_simulations, samples, columns, batch_size, epochs, test_size)
    trainer.run()
