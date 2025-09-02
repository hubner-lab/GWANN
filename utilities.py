from const import JSON_FILE, current_dict
import json 
from typing import Any
import numpy as np 
import tensorflow as tf
import os 
from mylogger import Logger


def json_update(key: str ,param:Any) -> None:
    """
    Update the JSON file with a new value for the specified key.

    Parameters:
    key (str): The key in the JSON file to update.
    param (int): The new value to set for the specified key.
    """
    tmp_dict = json.loads(JSON_FILE.read_text())
    tmp_dict[key] = param 
    JSON_FILE.write_text(json.dumps(tmp_dict))


def json_get(key:str):
    """get the value of the key in the data.json file"""
    if current_dict is None:
        tmp_dict = json.loads(JSON_FILE.read_text())
        return tmp_dict[key] if key in tmp_dict else None 
    return current_dict[key]


def createImages(columns, data, sim_indivduals):
    """
    Transform genomic data into images for CNN input, padding individuals if needed.
    Optimized for large datasets using vectorized operations and dask.

    Args:
        columns: Width of the output images.
        data: Input matrix from calc_avg_vcf, shape (n_variants, n_samples).
        sim_indivduals: Expected number of individuals (from json_get("samples")).

    Returns:
        TensorFlow tensor of images, shape (n_variants, rows, columns, 1).
    """

    logger = Logger('Message:', os.environ['LOGGER'])
    

    individuals = data.shape[1]
    logger.info(f"Input shape: {data.shape}")


    if individuals < sim_indivduals:
        pval = 0
        logger.info(f'Padding value:{pval}')
        pad = np.full((data.shape[0], sim_indivduals - individuals), pval) 
        data = np.concatenate((data, pad), axis=1)
        logger.info(f"Padded to shape: {data.shape}")

    

    data = data.astype(np.float32)


    rows = int(sim_indivduals / columns)
    logger.info(f"Image dimensions: ({rows}, {columns})")
   

    logger.info(f"Creating images with dimensions: ({rows}, {columns})")
    images = data.reshape(-1, rows, columns)  

    X = tf.convert_to_tensor(images, dtype=tf.float32)  
    X = tf.expand_dims(X, axis=-1) 
    logger.info(f"Output tensor shape: {X.shape}, dtype: {X.dtype}")

    return X


if __name__ == "__main__":
    print(json_get("SAMPLES"))
    json_update("SAMPLES", 100)
    print(json_get("SAMPLES"))  