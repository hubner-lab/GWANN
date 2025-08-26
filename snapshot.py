

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