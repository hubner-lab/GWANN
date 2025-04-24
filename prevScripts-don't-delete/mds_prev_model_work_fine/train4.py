from dataset4 import Dataset
from sklearn.model_selection import train_test_split 
import numpy as np
from const import MODEL_PATH_TENSOR_DIR
from mylogger import Logger
import os 
from keras.callbacks import ModelCheckpoint
from utilities import json_update, json_get
import tensorflow as tf
from net1 import ModelBuilder


class Train:
    def __init__(self, model_name:str, total_simulations:int, CausalSamples: int, columns:int,
                 batch_size:int, epochs:int, simPath: str = "./simulation/data/", test_ratio:float = 0.2):
        
        self.model_name = model_name
        self.dataset = Dataset(total_simulations, CausalSamples, columns, simPath)
        self.batch_size = batch_size
        self.epochs = epochs
        self.test_ratio = test_ratio
        self.logger = Logger(f'Message:', f"{os.environ['LOGGER']}")


    def data_splitter(self):
        return  train_test_split(
            self.dataset.X.numpy(),
            self.dataset.pop.numpy(),
            self.dataset.y.numpy(),
            test_size=self.test_ratio,
            random_state=42,
            stratify=self.dataset.y.numpy()
        )


    def run(self):
        self.logger.info("Starting training process...")
        self.logger.info(f"Loading data from {self.dataset.simPath}")
        self.dataset.load_data()
        self.logger.info("Data loaded successfully.") 
        self.logger.info("Splitting data into training and testing sets...")


        X_train, X_test, pop_train, pop_test, y_train, y_test = self.data_splitter()

        self.logger.debug(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")
        self.logger.info("Data split successfully.")

        pop_train = np.squeeze(pop_train, axis=1)
        pop_test = np.squeeze(pop_test, axis=1)
        samples = json_get("samples")



        self.height, self.width = self.dataset.X.shape[1:3]
        self.logger.info(f"Creating model with input shape: {self.height}x{self.width}x1")

        modelBuilder = ModelBuilder(self.height, self.width, self.dataset.pop.shape[2], samples)
        model = modelBuilder.model_summary()
        
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

        checkpoint_path = f"{MODEL_PATH_TENSOR_DIR}/{self.model_name}.h5"

        checkpoint = ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_accuracy',           # or 'val_accuracy' / 'val_acc'
                save_best_only=True,
                save_weights_only=False,
                verbose=1
                )

        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)

        train_dataset = tf.data.Dataset.from_tensor_slices(((X_train, pop_train), y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices(((X_test, pop_test), y_test))
        val_dataset = val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)


        self.logger.info("Training model...")

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
        self.logger.info(f"Model name updated to {MODEL_PATH_TENSOR_DIR}/{self.model_name}.h5")

        model.save(f"{MODEL_PATH_TENSOR_DIR}/{self.model_name}.h5")

        test_loss, test_acc = model.evaluate([X_test, pop_test], y_test)
        print('test loss', test_loss)
        print('test acc', test_acc)

        self.logger.info("Model training completed successfully.")

if __name__ == '__main__':
    total_simulations = 1000
    samples = 8
    columns = 20
    batch_size = 128
    epochs = 100
    test_size = 0.2

    trainer = Train(total_simulations, samples, columns, batch_size, epochs, test_size)
    trainer.run()
