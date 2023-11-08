import yaml

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Usage:
# config = load_config("conf/pipeline.yaml")
# ut_config = load_config("cicd/ut.yaml")
