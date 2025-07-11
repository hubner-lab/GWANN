from const import JSON_FILE, current_dict
import json 
from typing import Any
import numpy as np 
from genomeToImage import GenomeImage
from tqdm import tqdm
import tensorflow as tf
import glob
from scipy.special import logit


def tanh_map(output, scale=10):
    return np.tanh(scale * (output - 0.5))


def logit_map(output):
        EPSILON = 0.1
        clipoutput = np.clip(output, EPSILON, 1 - EPSILON)
        return logit(clipoutput)


def log_map(output):
    EPSILON  = 0.0001
    clipoutput = np.clip(output, EPSILON, 1 - EPSILON)  # to avoid log(1-1) = log(0)
    res =  -np.log(1-clipoutput)
    resNorm = res / np.max(res) # normalize to 0-1
    SCALE = 100
    return  SCALE* resNorm



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

def get_number_of_simulations(path):
    return len(glob.glob(f'{path}/**/genome*.txt', recursive=True))


def json_get(key:str):
    """get the value of the key in the data.json file"""
    if current_dict is None:
        tmp_dict = json.loads(JSON_FILE.read_text())
        return tmp_dict[key] if key in tmp_dict else None 
    return current_dict[key]



def createImages(columns, data, sim_indivduals, pad_value=-10):
    X_list = []
    individuals = data.shape[1]
    if individuals < sim_indivduals:
        pad = np.full((data.shape[0], sim_indivduals - individuals), pad_value)
        data = np.concatenate((data, pad), axis=1)
    elif individuals > sim_indivduals:
        data = data[:, :sim_indivduals]  # truncate - individuals))
        data = np.concatenate((data, pad), axis=1)
    rows = int(sim_indivduals / columns)
    genomeImage = GenomeImage(rows, columns)
    for sample in tqdm(data, desc="Generating images"):
        image = genomeImage.transform_to_image(sample)
        X_list.append(image)  # Append image to the list
    X = tf.convert_to_tensor(X_list, dtype=tf.float32)  # Adjust dtype if needed
    return X


if __name__ == "__main__":
    print(json_get("SAMPLES"))
    json_update("SAMPLES", 100)
    print(json_get("SAMPLES"))  