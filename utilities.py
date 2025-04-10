from const import JSON_FILE, current_dict
import json 
from typing import Any

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
        return tmp_dict[key] 
    return current_dict[key]




if __name__ == "__main__":
    print(json_get("SAMPLES"))
    json_update("SAMPLES", 100)
    print(json_get("SAMPLES"))  