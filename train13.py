from dataset4 import Dataset
from sklearn.model_selection import train_test_split 
import numpy as np
from const import MODEL_PATH_TENSOR_DIR
from mylogger import Logger
import os 
from utilities import json_update
from net3 import ModelBuilder  # assuming net3 is updated with softmax support
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from const import MODEL_NAME
class Train:
    def __init__(self, model_name:str, total_simulations:int, sampledSitesIncludeCausals: int, columns:int,
                 batch_size:int, epochs:int,mds:bool, numberOfIndividuals:int,
                 simPath: str = "./simulation/data/", test_ratio:float = 0.2):

        self.model_name = model_name
        self.dataset = Dataset(total_simulations, sampledSitesIncludeCausals,columns,mds, numberOfIndividuals ,simPath)
        self.batch_size = batch_size
        self.epochs = epochs
        self.test_ratio = test_ratio
        self.logger = Logger(f'Message:', f"{os.environ['LOGGER']}")

    def data_splitter(self):
        return train_test_split(
            self.dataset.X.numpy(),
            self.dataset.y.numpy(),
            test_size=self.test_ratio,
            random_state=42
        )

    def run(self):
        self.logger.info("Starting training process...")
        self.logger.info(f"Loading data from {self.dataset.simPath}")
        self.dataset.load_data()
        self.logger.info("Data loaded successfully.") 
        self.logger.info("Splitting data into training and testing sets...")

        X_train, X_test, y_train, y_test = self.data_splitter()

        self.logger.debug(f"y_train True labels: {len(y_train[y_train == 1]) }")
        self.logger.debug(f"y_train False labels: {len(y_train[y_train == 0]) }")
        self.logger.debug(f"y_test True labels: {len(y_test[y_test == 1]) }")
        self.logger.debug(f"y_test False labels: {len(y_test[y_test == 0]) }")

        y_train = to_categorical(y_train, 2)
        y_test = to_categorical(y_test, 2)

        self.logger.debug(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")
        self.logger.info("Data split successfully.")

        self.height, self.width = self.dataset.X.shape[1:3]
        self.logger.info(f"Creating model with input shape: {self.height}x{self.width}x1")

        modelBuilder = ModelBuilder(self.height, self.width)
        model = modelBuilder.model_summary()
        
        model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)

        self.logger.info("Training model...")

        checkpoint_cb = ModelCheckpoint(
            filepath=f"{MODEL_PATH_TENSOR_DIR}/{self.model_name}_best.h5",
            save_best_only=True,
            monitor='val_loss',  # or 'val_accuracy'
            mode='min',          # or 'max' for accuracy
            verbose=1
        )


        early_stopping_cb = EarlyStopping(
            monitor='val_loss',
            patience=30,         # You can adjust patience as needed
            restore_best_weights=True,
            verbose=1
        )

        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )

        model.fit(X_train, y_train, 
                validation_data=(X_test, y_test),
                  epochs=self.epochs, 
                  batch_size=self.batch_size, 
                  callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler],
                  verbose=1)

        json_update(MODEL_NAME, f"{MODEL_PATH_TENSOR_DIR}/{self.model_name}.h5")
        self.logger.info(f"Model name updated to {MODEL_PATH_TENSOR_DIR}/{self.model_name}.h5")
        model.save(f"{MODEL_PATH_TENSOR_DIR}/{self.model_name}.h5")


        test_loss, test_acc = model.evaluate(X_test, y_test)
        self.logger.info(f'Test loss: {test_loss}')
        self.logger.info(f'Test accuracy: {test_acc}')

        # Predict and calculate PR metrics
        y_pred_probs = model.predict(X_test)
        y_pred_labels = np.argmax(y_pred_probs, axis=1)
        y_test_labels = np.argmax(y_test, axis=1)

        precision = precision_score(y_test_labels, y_pred_labels)
        recall = recall_score(y_test_labels, y_pred_labels)
        f1 = f1_score(y_test_labels, y_pred_labels)
        auc_pr = average_precision_score(y_test[:, 1], y_pred_probs[:, 1])

        self.logger.info(f'Precision: {precision:.4f}')
        self.logger.info(f'Recall: {recall:.4f}')
        self.logger.info(f'F1-score: {f1:.4f}')
        self.logger.info(f'AUC-PR (Precision-Recall): {auc_pr:.4f}')
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
