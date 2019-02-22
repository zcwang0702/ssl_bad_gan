import os


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_instance(module, name, config, *args):
    # first get module using getattr and then assign config parameters
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])
