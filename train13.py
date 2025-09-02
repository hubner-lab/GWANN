from dataset4 import Dataset
import numpy as np
from const import MODEL_PATH_TENSOR_DIR
from mylogger import Logger
import os 
from utilities import json_update
from net5 import ModelBuilder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from TrainingVisualizer import TrainingVisualizer
from sklearn.metrics import matthews_corrcoef
from tensorflow.keras import backend as K


def f1_m(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    return K.mean(f1)


class Train:
    
    def __init__(self, model_name:str, total_simulations:int,
                sampledSitesIncludeCausals: int, columns:int,
                batch_size:int = 64, learning_rate:float = 0.01,
                epochs:int = 1000,mds:bool = False, 
                simPath:str = "./simulation/data/",test_ratio:float = 0.2
                ):

        self.model_name = model_name
        self.learning_rate = learning_rate
        self.dataset = Dataset(total_simulations, sampledSitesIncludeCausals,columns,mds, simPath)
        self.batch_size = batch_size
        self.epochs = epochs
        self.test_ratio = test_ratio
        self.logger = Logger(f'Message:', f"{os.environ['LOGGER']}")
    

    def shuffle_data(self, X, y):
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        return X[idx], y[idx]
    
    
    def data_splitter(self):
        tp_indices = np.where(self.dataset.y.numpy() == 1)
        tn_indices = np.where(self.dataset.y.numpy() == 0)
        self.logger.debug(f"Total samples: {len(self.dataset.y)}")

        tp_images = self.dataset.X.numpy()[tp_indices]
        tn_images = self.dataset.X.numpy()[tn_indices]
        self.logger.debug(f"Number of True Positive samples: {len(tp_images)}")
        self.logger.debug(f"Number of True Negative samples: {len(tn_images)}")

        cut_train_tp = int((1 - self.test_ratio) * len(tp_images))
        cut_train_tn = int((1 - self.test_ratio) * len(tn_images))
        self.logger.debug(f"Cut points for training: {cut_train_tp} TP, {cut_train_tn} TN")

        # Training sets
        X_train_tp = tp_images[:cut_train_tp]
        X_train_tn = tn_images[:cut_train_tn]
        Y_train_tp = np.ones(len(X_train_tp), dtype=int)
        Y_train_tn = np.zeros(len(X_train_tn), dtype=int)
        self.logger.debug(f"Training set before shuffle: {len(X_train_tp)} TP, {len(X_train_tn)} TN")

        # Remaining data for validation + test
        X_rest_tp = tp_images[cut_train_tp:]
        X_rest_tn = tn_images[cut_train_tn:]
        self.logger.debug(f"Remaining for val/test: {len(X_rest_tp)} TP, {len(X_rest_tn)} TN")

        validation_ratio = 0.2
        cut_val_tp = int(validation_ratio * len(X_rest_tp))
        cut_val_tn = int(validation_ratio * len(X_rest_tn))
        self.logger.debug(f"Cut points for validation: {cut_val_tp} TP, {cut_val_tn} TN")

        # Validation sets
        X_val_tp = X_rest_tp[:cut_val_tp]
        X_val_tn = X_rest_tn[:cut_val_tn]
        self.logger.debug(f"Validation set before shuffle: {len(X_val_tp)} TP, {len(X_val_tn)} TN")

        # Test sets
        X_test_tp = X_rest_tp[cut_val_tp:]
        X_test_tn = X_rest_tn[cut_val_tn:]
        self.logger.debug(f"Test set before shuffle: {len(X_test_tp)} TP, {len(X_test_tn)} TN")

        # Merge and shuffle
        X_train, y_train = self.shuffle_data(
            np.concatenate((X_train_tp, X_train_tn), axis=0),
            np.concatenate((Y_train_tp, Y_train_tn), axis=0)
        )
        self.logger.debug(f"Training set after shuffle: {len(X_train)} samples")

        X_val, y_val = self.shuffle_data(
            np.concatenate((X_val_tp, X_val_tn), axis=0),
            np.concatenate((np.ones(len(X_val_tp), dtype=int), np.zeros(len(X_val_tn), dtype=int)), axis=0)
        )
        self.logger.debug(f"Validation set after shuffle: {len(X_val)} samples")

        X_test, y_test = self.shuffle_data(
            np.concatenate((X_test_tp, X_test_tn), axis=0),
            np.concatenate((np.ones(len(X_test_tp), dtype=int), np.zeros(len(X_test_tn), dtype=int)), axis=0)
        )
        self.logger.debug(f"Test set after shuffle: {len(X_test)} samples")

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

        
        y_train = to_categorical(y_train, 2)
        y_test = to_categorical(y_test, 2)
        y_val = to_categorical(y_val, 2)

        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)
        X_val = np.expand_dims(X_val, -1)

        self.logger.info("Data split successfully.")

        self.height, self.width = self.dataset.X.shape[1:3]
        self.logger.info(f"Creating model with input shape: {self.height}x{self.width}x1")

        modelBuilder = ModelBuilder(self.height, self.width)
        model = modelBuilder.model_summary()

        self.logger.info(f'Setting model learning rate to: {self.learning_rate}')
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='categorical_crossentropy', metrics=['accuracy', f1_m])

     
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
                  verbose=1)

        json_update("model_name", f"{MODEL_PATH_TENSOR_DIR}/{self.model_name}.h5")
        self.logger.info(f"Model name updated to {MODEL_PATH_TENSOR_DIR}/{self.model_name}.h5")
        model.save(f"{MODEL_PATH_TENSOR_DIR}/{self.model_name}.h5")


        test_loss, test_acc, test_f1  = model.evaluate(X_test, y_test)
        self.logger.info(f'Test loss: {test_loss}')
        self.logger.info(f'Test accuracy: {test_acc}')
        self.logger.info(f'Test F1-score: {test_f1}')   
  
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

        mcc = matthews_corrcoef(y_test_labels, y_pred_labels)
        self.logger.info(f'Matthews Correlation Coefficient: {mcc:.4f}')
        visualizer = TrainingVisualizer(f'./metrics/{self.batch_size}_{self.learning_rate}_{self.dataset.sampledSitesIncludeCausals}')  
        visualizer.plot_confusion_matrix(y_test_labels, y_pred_labels)
        visualizer.plot_precision_recall(y_test[:, 1], y_pred_probs[:, 1]) 
        visualizer.plot_roc_curve(y_test[:, 1], y_pred_probs[:, 1])
        visualizer.plot_f1_score(history)
        visualizer.plot_full_history(history)


if __name__ == '__main__':
    total_simulations = 1000
    samples = 8
    columns = 20
    batch_size = 128
    epochs = 100
    test_size = 0.2

    trainer = Train(total_simulations, samples, columns, batch_size, epochs, test_size)
    trainer.run()
