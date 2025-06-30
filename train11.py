from dataset4 import Dataset
from sklearn.model_selection import train_test_split 
import numpy as np
from const import MODEL_PATH_TENSOR_DIR
from mylogger import Logger
import os 
from utilities import json_update
from net3 import ModelBuilder  # assuming net3 is updated with softmax support
# from net3_BN import ModelBuilder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.callbacks import ReduceLROnPlateau

class Train:
    def __init__(self, model_name: str, total_simulations: int, sampledSitesIncludeCausals: int, columns: int,
                 batch_size: int, epochs: int, mds: bool , simPath: str = "./simulation/data/", test_ratio: float = 0.2):

        self.model_name = model_name
        self.dataset = Dataset(total_simulations, sampledSitesIncludeCausals, columns, mds, simPath)
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

        self.logger.debug(f"y_train True labels: {len(y_train[y_train == 1])}")
        self.logger.debug(f"y_train False labels: {len(y_train[y_train == 0])}")
        self.logger.debug(f"y_test True labels: {len(y_test[y_test == 1])}")
        self.logger.debug(f"y_test False labels: {len(y_test[y_test == 0])}")

        y_train = to_categorical(y_train, 2)
        y_test = to_categorical(y_test, 2)

        self.logger.debug(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")
        self.logger.info("Data split successfully.")

        self.height, self.width = self.dataset.X.shape[1:3]
        self.logger.info(f"Creating model with input shape: {self.height}x{self.width}x1")

        modelBuilder = ModelBuilder(self.height, self.width)
        model = modelBuilder.model_summary()

        model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')]
        )

        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)

        total_causal = len(self.dataset.y[self.dataset.y == 1])
        total_non_causal_mds_include = len(self.dataset.y[self.dataset.y == 0])
        # ratio = int(total_non_causal_mds_include / total_causal)
        # class_weight_dict = {0: 1.0, 1: 1.0}
        # self.logger.info(f"Using class weights: {class_weight_dict}")

        self.logger.info("Training model...")

        patience = 100
        max_retries = 100
        retries = 0
        # best_val_auc = float('-inf')
        best_val_precision = float('-inf')
        while retries <= max_retries:
            checkpoint_cb = ModelCheckpoint(
                filepath=f"{MODEL_PATH_TENSOR_DIR}/{self.model_name}_best.h5",
                save_best_only=True,
                monitor='val_precision',
                mode='max',
                verbose=1
            )

            reduce_lr_cb = ReduceLROnPlateau(
                monitor='val_precision',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )

            early_stop_cb = EarlyStopping(
                monitor='val_precision',
                patience=patience,
                restore_best_weights=True,
                mode='max',
                verbose=1
            )

            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb],
                # class_weight=class_weight_dict,
                verbose=1
            )

            # val_auc = max(history.history['val_auc'])
            val_precision = max(history.history['val_precision'])
            if val_precision > best_val_precision:
                best_val_precision = val_precision
                break
            else:
                retries += 1
                self.logger.warning(f"No improvement in val_auc. Restarting training... Attempt {retries}/{max_retries}")
                model = modelBuilder.model_summary()
                model.compile(
                    optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')]
                )

        json_update("model_name", f"{MODEL_PATH_TENSOR_DIR}/{self.model_name}.h5")
        self.logger.info(f"Model name updated to {MODEL_PATH_TENSOR_DIR}/{self.model_name}.h5")
        model.save(f"{MODEL_PATH_TENSOR_DIR}/{self.model_name}.h5")

        test_loss, test_acc, test_auc, test_precision, test_recall = model.evaluate(X_test, y_test)
        self.logger.info(f'Test loss: {test_loss}')
        self.logger.info(f'Test accuracy: {test_acc}')
        self.logger.info(f'Test AUC: {test_auc}')
        self.logger.info(f'Test Precision: {test_precision}')
        self.logger.info(f'Test Recall: {test_recall}')
        self.logger.info("Model training completed successfully.")

if __name__ == '__main__':
    total_simulations = 1000
    samples = 8
    columns = 20
    batch_size = 128
    epochs = 100
    test_size = 0.2

    trainer = Train("model_tp_weighted", total_simulations, samples, columns, batch_size, epochs, test_size)
    trainer.run()
