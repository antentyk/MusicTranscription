import json
import os

def __get_config():
    """
    Returns dictionary of configurations stored in .json files
    in config folder in the root of the project

    The config itself contains path to MAPS database,
    data representation parameters and other information

    Returns:
        dict: configuration dictionary
    """

    config_filenames = filter(lambda x: x.endswith("_config.json"), os.listdir("./config"))
    config_filenames = map(lambda x: "./config/" + x, config_filenames)

    configuration = {}

    for filename in config_filenames:
        with open(filename) as file:
            configuration.update(json.loads(file.read()))
    
    return configuration

config = __get_config()
