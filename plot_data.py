import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

PATHS = ['/mnt/data/amir/GWANN-TEST/GWANN/metrics/1M-DATA-SET/64_0.01_50',
         '/mnt/data/amir/GWANN-TEST/GWANN/metrics/1M-DATA-SET/64_0.01_60',
         '/mnt/data/amir/GWANN-TEST/GWANN/metrics/1M-DATA-SET/64_0.01_70',
         '/mnt/data/amir/GWANN-TEST/GWANN/metrics/1M-DATA-SET/64_0.01_130',
         "/mnt/data/amir/GWANN-TEST/GWANN/metrics/1M-DATA-SET/64_0.01_120",
         '/mnt/data/amir/GWANN-TEST/GWANN/metrics/1M-DATA-SET/64_0.01_140',
         '/mnt/data/amir/GWANN-TEST/GWANN/metrics/1M-DATA-SET/64_0.01_160',
         '/mnt/data/amir/GWANN-TEST/GWANN/metrics/1M-DATA-SET/64_0.01_190',
         ]


def get_images(path):
    images = os.listdir(path)
    return [os.path.join(path, img) for img in images if img.endswith('.png')]


def build_figure(MATRIX):
    fig, ax = plt.subplots(len(MATRIX), 5, figsize=(20, 20))

    for i, path in enumerate(MATRIX):
        images = get_images(path)
        for j, img_path in enumerate(images):
            img = Image.open(img_path)
            ax[i, j].imshow(np.array(img))
            ax[i, j].axis('off')
            ax[i, j].set_title(os.path.basename(img_path))
    plt.tight_layout()
    plt.savefig('output_figure.png', dpi=300)

build_figure(PATHS)
