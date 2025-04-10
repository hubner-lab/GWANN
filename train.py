from dataset import Dataset
from sklearn.model_selection import train_test_split 
from tensorflow import keras
from keras import layers
import numpy as np
from const import MODEL_PATH_TENSOR_DIR
from mylogger import Logger
import os 
from utilities import json_update
class Train:
    def __init__(self,model_name:str, total_simulations:int, CausalSamples: int, columns:int ,
                  batch_size:int , epochs:int, simPath: str = "./simulation/data/", test_ratio:float=0.2):
        self.model_name = model_name
        self.dataset = Dataset(total_simulations, CausalSamples, columns, simPath)
        self.batch_size = batch_size
        self.epochs = epochs
        self.test_ratio = test_ratio
    def run(self):
        Logger(f'Message:', os.environ['LOGGER']).info("Starting training process...")
        Logger(f'Message:', os.environ['LOGGER']).info(f"Loading data from {self.dataset.simPath}")
        self.dataset.load_data()
        # individuals = self.dataset.X.shape[1] * self.dataset.X.shape[2] 
        Logger(f'Message:', os.environ['LOGGER']).info("Data loaded successfully.") 
        Logger(f'Message:', os.environ['LOGGER']).info("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            self.dataset.X.numpy(),  # Convert tensor to numpy array for splitting
            self.dataset.y.numpy(),  # Convert tensor to numpy array for splitting
            test_size=self.test_ratio,  # Percentage of data for testing
            random_state=42  # For reproducibility
        )
        Logger(f'Message:', os.environ['LOGGER']).debug(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")
        Logger(f'Message:', os.environ['LOGGER']).info("Data split successfully.")


        self.height, self.width = self.dataset.X.shape[1:3] 
        Logger(f'Message:', os.environ['LOGGER']).info(f"Creating model with input shape: {self.height}x{self.width}x1")
        C1, C2, C3 = 2, 2, 2  # Kernel sizes
        P = 2  # Pooling size

        
        inputs = keras.Input(shape=(self.height, self.width, 1))

        # Convolutional layers
        x = layers.Conv2D(3,(C1, C1), activation='relu')(inputs)
        x = layers.MaxPooling2D(pool_size=(P, P))(x)
        x = layers.Conv2D(5, (C2, C2), activation='relu')(x)
        x = layers.MaxPooling2D(pool_size=(P, P))(x)
        x = layers.Conv2D(10, (C3, C3), activation='relu')(x)
        x = layers.Dense(10)(x)
        x = layers.Dropout(rate=0.2)(x)
        x = layers.Flatten()(x)
        output =layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(inputs=inputs, outputs=output)

        model.summary()
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


        X_train = np.expand_dims(X_train, -1)  # include chanel dim
        X_test = np.expand_dims(X_test, -1)  # include chanel dim
        Logger(f'Message:', os.environ['LOGGER']).info("Training model...") 
        model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

        json_update("model_name", f"{MODEL_PATH_TENSOR_DIR}/{self.model_name}.h5")
        Logger(f'Message:',f"Model name updated to {MODEL_PATH_TENSOR_DIR}/{self.model_name}.h5")
        model.save(f"{MODEL_PATH_TENSOR_DIR}/{self.model_name}.h5")  # Saves the model as a .h5 file
        
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print('test loss', test_loss)
        print('test acc', test_acc)
        Logger(f'Message:', os.environ['LOGGER']).info("Model training completed successfully.")    


if __name__ == '__main__':
    total_simulations = 1000
    samples = 8
    columns = 20
    batch_size = 128
    epochs = 100
    test_size = 0.2

    trainer = Train(total_simulations, samples, columns, batch_size, epochs, test_size)
    trainer.run()