import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, roc_curve, auc
)


class TrainingVisualizer:
    def __init__(self, save_dir="./"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def plot_history(self, history):
        plt.figure(figsize=(12, 5))

        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/training_history.png")
        plt.close()


    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.savefig(f"{self.save_dir}/confusion_matrix.png")
        plt.close()


    def plot_precision_recall(self, y_true, y_probs):
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        plt.figure()
        plt.plot(recall, precision, label='PR Curve')
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{self.save_dir}/pr_curve.png")
        plt.close()


    def plot_roc_curve(self, y_true, y_probs):
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title('Receiver Operating Characteristic')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{self.save_dir}/roc_curve.png")
        plt.close()
        
        
    def plot_learning_rate(self, history):
        if 'lr' not in history.history:
            print("Learning rate history not found in training history.")
            return

        plt.figure()
        plt.plot(history.history['lr'], label='Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{self.save_dir}/learning_rate.png")
        plt.close()
    
    
    def plot_f1_score(self, history):
        if 'f1_m' not in history.history and 'val_f1_m' not in history.history:
            print("F1 score not found in training history.")
            return

        plt.figure()
        plt.plot(history.history.get('f1_m', []), label='Train F1')
        plt.plot(history.history.get('val_f1_m', []), label='Val F1')
        plt.title('F1 Score Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{self.save_dir}/f1_score.png")
        plt.close()


    def plot_full_history(self, history):
        has_lr = 'lr' in history.history
        n_subplots = 3 if has_lr else 2

        plt.figure(figsize=(15, 5))

        # Accuracy
        plt.subplot(1, n_subplots, 1)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        # Loss
        plt.subplot(1, n_subplots, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Learning Rate (if available)
        if has_lr:
            plt.subplot(1, n_subplots, 3)
            plt.plot(history.history['lr'], label='Learning Rate')
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('LR')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/full_training_history.png")
        plt.close()