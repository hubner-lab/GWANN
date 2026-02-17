
import numpy as np
import matplotlib.pyplot as plt
import os 


class GenomeImage:
    def __init__(self,  rows:int, colums:int):
        self.rows = rows
        self.colums = colums
    

    def transform_to_image(self, data:np.array):
        return data.reshape(self.rows, self.colums)
    
    
    def plot_sample(self, dir_name,sample, image_name):
        plt.imshow(sample,  cmap="gray", vmin=0,vmax=1) 
        plt.title(image_name) 
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        plt.savefig(f"{dir_name}/{image_name}.png")  
        #plt.show() 
        plt.close()  


