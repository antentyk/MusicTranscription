import json

def get_config():
    """
    Returns dictionary of configurations stored in .json files
    in config folder in the root of the project

    The config itself contains path to MAPS database,
    data representation parameters and other information

    Returns:
        dict: configuration dictionary
    """
    config_filenames = [
        "config/representation.json",
        "config/local_config.json"
    ]

    config = {}

    for filename in config_filenames:
        with open(filename) as file:
            config.update(json.loads(file.read()))
    
    return config

config = get_config()
