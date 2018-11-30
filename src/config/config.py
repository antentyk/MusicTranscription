import json

def get_config():
    config_filenames = [
        "../../config/representation.json",
        "../../config/local_config.json"
    ]

    config = {}

    for filename in config_filenames:
        with open(filename) as file:
            config.update(json.loads(file.read()))
    
    return config

config = get_config()
