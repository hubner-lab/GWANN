from const import JSON_FILE, current_dict
import json 
from typing import Any
import numpy as np 
from genomeToImage import GenomeImage
from tqdm import tqdm
import tensorflow as tf

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
    X_list = []
    individuals = data.shape[1]
    if individuals  < sim_indivduals:
        pad = np.zeros((data.shape[0], sim_indivduals - individuals))
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