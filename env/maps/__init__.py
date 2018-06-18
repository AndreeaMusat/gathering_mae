import os


def get_map_path(name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(dir_path, name + ".txt")
    assert os.path.isfile(path), "Map file not found: {}".format(name)
    return path

