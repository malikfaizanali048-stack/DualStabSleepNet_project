"""
Single shared config loader for the whole project. Recursively merges a
`defaults:` parent chain (base.yaml -> sleepedf78.yaml -> _test_synthetic.yaml
etc) so overriding e.g. optim.batch_size in a child config does NOT wipe out
sibling keys like optim.lr from the parent (a shallow {**parent, **child}
merge would do exactly that -- this was a real bug caught during smoke
testing, fixed here once, used everywhere).
"""
import os

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if "defaults" in cfg:
        parent_path = os.path.join(os.path.dirname(path), cfg.pop("defaults"))
        parent_cfg = load_config(parent_path)  # recursive: handles multi-level chains
        cfg = deep_merge(parent_cfg, cfg)
    return cfg