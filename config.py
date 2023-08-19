import sys
import yaml
import glob
import os
import math
try:
    sys.path.append(glob.glob('.../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


def cfg_from_yaml_file(cfg_file):
    with open(cfg_file, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config

def config_to_trans(trans_config):
    transform = carla.Transform(carla.Location(trans_config["location"][0],
                                               trans_config["location"][1],
                                               trans_config["location"][2]),
                                carla.Rotation(trans_config["rotation"][0],
                                               trans_config["rotation"][1],
                                               trans_config["rotation"][2]))
    return transform

def cfg_from_list(cfg, set_list):
    """Set config keys via list (e.g., from command line)."""
    if set_list is None:
        return
    from ast import literal_eval
    assert  len(set_list) % 2 == 0
    for k, v in zip(set_list[0::2], set_list[1::2]):
        key_list = k.split('.')
        d = cfg
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), 'type {} does not match original type {}'.format(type(value), type(d[subkey]))
        d[subkey] = value
        
        
def get_agent_names(cfg):
    if cfg["AGENT_CONFIG"] is None:
        return []
    result = []
    for agent_name, agent_config in cfg["AGENT_CONFIG"].items():
        if agent_config["ENABLE"] and "SENSOR_CONFIG" in agent_config:
            result.append(agent_name)
    return result

def draw_rectangle(o_x, o_y, o_z, L, W, x):
    col_num = math.ceil(L / x) + 1
    row_num = math.ceil(W / x) + 1
    
    result_list = []
    for i in range(1, col_num+1):
        for j in range(1, row_num+1):
            fill_x = o_x - L/2 + (i-1) * x
            fill_y = o_y + W/2 - (j-1) * x
            fill_point = (fill_x, fill_y)
            if fill_point != (o_x, o_y):
                result_list.append([round(fill_x, 2), round(fill_y, 2), o_z])
    return result_list


def split_file_name(text):
    file_path, file_name = os.path.split(text)
    short_name, extension = os.path.splitext(file_name)  
    return file_path, short_name, extension