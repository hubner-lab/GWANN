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
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from TrainingVisualizer import TrainingVisualizer


class Train:
    def __init__(self, model_name:str, total_simulations:int, sampledSitesIncludeCausals: int, columns:int,
                 batch_size:int, epochs:int,mds:bool, simPath: str = "./simulation/data/", test_ratio:float = 0.2):

        self.model_name = model_name
        self.dataset = Dataset(total_simulations, sampledSitesIncludeCausals,columns,mds, simPath)
        self.batch_size = batch_size
        self.epochs = epochs
        self.test_ratio = test_ratio
        self.logger = Logger(f'Message:', f"{os.environ['LOGGER']}")
    

    def data_splitter(self):
        # First split into train and temp (which will be split into val and test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.dataset.X.numpy(),
            self.dataset.y.numpy(),
            test_size=self.test_ratio, 
            random_state=42
        )

        val_ratio = 0.5  
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=val_ratio,
            random_state=42
        )

        return X_train, X_val, X_test, y_train, y_val, y_test


    def run(self):
        self.logger.info("Starting training process...")
        self.logger.info(f"Loading data from {self.dataset.simPath}")
        self.dataset.load_data()
        self.logger.info("Data loaded successfully.") 
        self.logger.info("Splitting data into training and testing sets...")

        X_train, X_val, X_test, y_train, y_val, y_test = self.data_splitter()

        self.logger.debug(f"y_train True labels: {len(y_train[y_train == 1]) }")
        self.logger.debug(f"y_train False labels: {len(y_train[y_train == 0]) }")
        self.logger.debug(f"y_test True labels: {len(y_test[y_test == 1]) }")
        self.logger.debug(f"y_test False labels: {len(y_test[y_test == 0]) }")
        self.logger.debug(f"y_val True labels: {len(y_val[y_val == 1]) }")
        self.logger.debug(f"y_val False labels: {len(y_val[y_val == 0]) }")


        class_weights = {0:1, 1:1}

        self.logger.info(f"Class weights: {class_weights}")
        
        y_train = to_categorical(y_train, 2)
        y_test = to_categorical(y_test, 2)
        y_val = to_categorical(y_val, 2)

        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)
        X_val = np.expand_dims(X_val, -1)

        self.logger.debug(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")
        self.logger.info("Data split successfully.")

        self.height, self.width = self.dataset.X.shape[1:3]
        self.logger.info(f"Creating model with input shape: {self.height}x{self.width}x1")

        modelBuilder = ModelBuilder(self.height, self.width)
        model = modelBuilder.model_summary()
        
        model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

     
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

        history = model.fit(X_train, y_train, 
                validation_data=(X_val, y_val),
                  epochs=self.epochs, 
                  batch_size=self.batch_size, 
                  callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler],
                #   class_weight=class_weights,
                  verbose=1)

        json_update("model_name", f"{MODEL_PATH_TENSOR_DIR}/{self.model_name}.h5")
        self.logger.info(f"Model name updated to {MODEL_PATH_TENSOR_DIR}/{self.model_name}.h5")
        model.save(f"{MODEL_PATH_TENSOR_DIR}/{self.model_name}.h5")


        test_loss, test_acc = model.evaluate(X_test, y_test)
        self.logger.info(f'Test loss: {test_loss}')
        self.logger.info(f'Test accuracy: {test_acc}')

  
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

        visualizer = TrainingVisualizer('./metrics')  
        visualizer.plot_history(history) 
        visualizer.plot_confusion_matrix(y_test_labels, y_pred_labels)
        visualizer.plot_precision_recall(y_test[:, 1], y_pred_probs[:, 1]) 
        visualizer.plot_roc_curve(y_test[:, 1], y_pred_probs[:, 1])


if __name__ == '__main__':
    total_simulations = 1000
    samples = 8
    columns = 20
    batch_size = 128
    epochs = 100
    test_size = 0.2

    trainer = Train(total_simulations, samples, columns, batch_size, epochs, test_size)
    trainer.run()
