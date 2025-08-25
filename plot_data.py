import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from selenium import webdriver
from pathlib import Path
import argparse
class Screenshot:
    def __init__(self, url, output_path):
        self.url = Path(url).absolute().as_uri()
        self.output_path = output_path
    def take_screenshot(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)
        driver.get(self.url)
        driver.save_screenshot(self.output_path)
        driver.quit()




class Filter:
    def __init__(self, baseDirPath):
        self.path = baseDirPath

    def filter(self, condition):
        return [
            os.path.join(self.path, f)
            for f in os.listdir(self.path)
            if condition(f)
        ]
    
    def _safe_float(self, value):
        try:
            return float(value)
        except ValueError:
            return float('inf')
        
    def filter_by_parts(self, first=None, middle=None, last=None):
        filtered = self.filter(lambda name: self._matches_parts(name, first, middle, last))
        return sorted(filtered, key=lambda p: self._safe_float(os.path.basename(p).split("_")[-1]))

    def _matches_parts(self, name, first, middle, last):
        parts = name.split("_")
        if len(parts) != 3:
            return False
        return (
            (first is None or parts[0] == str(first)) and
            (middle is None or parts[1] == str(middle)) and
            (last is None or parts[2] == str(last))
        )



class FigureBuilder:
    def __init__(self, paths, output_dir='output'):
        self.paths = paths
        self.output_dir = output_dir

    def build_figures(self):
        fig, ax = plt.subplots(len(self.paths), 6, figsize=(20, 20))

        for i, path in enumerate(self.paths):
            self._create_image_from_html(path)
            images = self.get_images(path)
            for j, img_path in enumerate(images):
                img = Image.open(img_path)
                ax[i, j].imshow(np.array(img))
                ax[i, j].axis('off')
                ax[i, j].set_title(os.path.basename(img_path))
        plt.tight_layout()
        plt.savefig(self.output_dir, dpi=300)

    def _create_image_from_html(self, path):
        images = os.listdir(path)
        html_file = [os.path.join(path, img) for img in images if img.endswith('.html')][0]
        parts = os.path.basename(path).split("_")
        formatted_name = f"Batch={parts[0]}_LR={parts[1]}_SR={parts[2]}.png"
        if formatted_name not in images:
            Screenshot(html_file, os.path.join(path, formatted_name)).take_screenshot()

    def get_images(self, path):
        images = os.listdir(path)
        return [os.path.join(path, img) for img in images if img.endswith('.png')]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build figure from filtered experiment results.")
    parser.add_argument("--b", type=str, default=None, help="First part of directory name (e.g., batch size).")
    parser.add_argument("--lr", type=str, default=None, help="Middle part of directory name (e.g., learning rate).")
    parser.add_argument("--sr", type=str, default=None, help="Last part of directory name (e.g., sample rate).")
    parser.add_argument("--p", type=str, required=True, help="Base directory path containing results.")
    args = parser.parse_args()


    f = Filter(args.p)
    FigureBuilder(f.filter_by_parts(first=args.b, middle=args.lr, last=args.sr), 
                  f'Batch={args.b if args.b else "all"},LR={args.lr if args.lr else "all"},SR={args.sr if args.sr else "all"}.png').build_figures()
