

import numpy as np
import os
from PIL import Image
from genomeToImage import GenomeImage
class SnapShot:
    def __init__(self, X, labels, save_dir):
        self.X = X
        self.labels = labels
        self.save_dir = save_dir
        self.genome_image = GenomeImage(1,1)  # Dummy initialization
    
    def _get_tp_samples(self):
        SAMPLES = 100
        tp_indices = np.where(self.labels == 1)[0]
        tp_samples = self.X[tp_indices]
        return tp_samples
    


    def save_snapshot(self, filename: str) -> None:
        """
        Save a snapshot of true positive samples to a .npz file.

        Parameters:
        filename (str): The name of the file to save the snapshot to.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        for i, sample in enumerate(self._get_tp_samples()):
            file_path = os.path.join(self.save_dir, f"{filename}_sample_{i}.npy")
            np.save(file_path, sample)
            self.genome_image.plot_sample(self.save_dir,sample, f"{filename}_sample_{i}")
    
    def save_prediction(self, filename: str) -> None:
        """
        Save a random snapshot of up to 100 true positive samples to .npy files and images.

        Parameters:
        filename (str): The name of the file to save the snapshot to.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        tp_samples = self._get_tp_samples()
        SAMPLES = 100

        # If more than 100, randomly choose 100 without replacement
        if len(tp_samples) > SAMPLES:
            indices = np.random.choice(len(tp_samples), SAMPLES, replace=False)
            tp_samples = tp_samples[indices]

        for i, sample in enumerate(tp_samples):
            file_path = os.path.join(self.save_dir, f"{filename}_sample_{i}.npy")
            np.save(file_path, sample)
            self.genome_image.plot_sample(self.save_dir, sample[:, :, :, 0], f"{filename}_sample_{i}")
