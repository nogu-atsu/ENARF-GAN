import yaml
from easydict import EasyDict as edict


def yaml_config(config, default_cofig, resume_latest=False, num_workers=1):
    default = edict(yaml.load(open(default_cofig), Loader=yaml.SafeLoader))
    conf = edict(yaml.load(open(config), Loader=yaml.SafeLoader))

    def copy(conf, default):
        for key in conf:
            if isinstance(default[key], edict):
                copy(conf[key], default[key])
            else:
                default[key] = conf[key]

    copy(conf, default)

    default.resume_latest = resume_latest
    default.dataset.num_workers = num_workers
    return default
