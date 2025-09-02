

import numpy as np
import os
from genomeToImage import GenomeImage
class SnapShot:
    def __init__(self, X, labels, save_dir, creat_numpy=False):
        self.X = X
        self.labels = labels
        self.save_dir = save_dir
        self.createt_numpy = creat_numpy
        self.genome_image = GenomeImage(1,1)  # Dummy initialization
    

    def _get_samples(self, label=1):
        tp_indices = np.where(self.labels == label)[0]
        tp_samples = self.X[tp_indices]
        return tp_samples
    

    def save_snapshot_tp(self, filename: str) -> None:
        """
        Save a snapshot of true positive samples to a .npz file.

        Parameters:
        filename (str): The name of the file to save the snapshot to.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        for i, sample in enumerate(self._get_samples()):
            if self.createt_numpy:
                file_path = os.path.join(self.save_dir, f"{filename}_sample_{i}.npy")
                np.save(file_path, sample)
            self.genome_image.plot_sample(self.save_dir,sample, f"{filename}_sample_{i}")
    
    
    def save_prediction_tp(self, filename: str) -> None:
        """
        Save a random snapshot of up to 100 true positive samples to .npy files and images.

        Parameters:
        filename (str): The name of the file to save the snapshot to.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        tp_samples = self._get_samples()
        SAMPLES = 100

        if len(tp_samples) > SAMPLES:
            indices = np.random.choice(len(tp_samples), SAMPLES, replace=False)
            tp_samples = tp_samples[indices]

        for i, sample in enumerate(tp_samples):
            if self.createt_numpy:
                file_path = os.path.join(self.save_dir, f"{filename}_sample_{i}.npy")
                np.save(file_path, sample)
            self.genome_image.plot_sample(self.save_dir, sample[:, :, :, 0], f"{filename}_sample_{i}")

    
    def save_snapshot_fp(self, filename: str) -> None:
        """
        Save a snapshot of true negative samples to a .npz file.

        Parameters:
        filename (str): The name of the file to save the snapshot to.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        SAMPLES = 3

        tp_samples = self._get_samples(label=0)
        if len(tp_samples) > SAMPLES:
            indices = np.random.choice(len(tp_samples), SAMPLES, replace=False)
            tp_samples = tp_samples[indices]

        for i, sample in enumerate(tp_samples):
            if self.createt_numpy:
                file_path = os.path.join(self.save_dir, f"{filename}_sample_{i}.npy")
                np.save(file_path, sample)
            self.genome_image.plot_sample(self.save_dir,sample, f"{filename}_sample_{i}")
    

    def save_prediction_fp(self, filename: str) -> None:
        """
        Save a random snapshot of up to 100 true negative samples to .npy files and images.

        Parameters:
        filename (str): The name of the file to save the snapshot to.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        tp_samples = self._get_samples(label=0)
        SAMPLES = 100

        if len(tp_samples) > SAMPLES:
            indices = np.random.choice(len(tp_samples), SAMPLES, replace=False)
            tp_samples = tp_samples[indices]

        for i, sample in enumerate(tp_samples):
            if self.createt_numpy:
                file_path = os.path.join(self.save_dir, f"{filename}_sample_{i}.npy")
                np.save(file_path, sample)
            self.genome_image.plot_sample(self.save_dir, sample[:, :, :, 0], f"{filename}_sample_{i}")