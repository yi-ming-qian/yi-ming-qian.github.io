from config.config_wifi import WifiConfig


def get_config(name):
    if name == 'wifi':
        return WifiConfig
    else:
        raise ValueError("Got config name: {}".format(name))
