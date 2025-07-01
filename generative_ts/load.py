import params, data, model
from params import DEVICE, BASE_DIR, SAVE_DIR

import os
import json
import copy

import torch


def get_cfg(target = None):

    load_path = os.path.exists(os.path.join(SAVE_DIR, target)) or sorted(os.listdir(SAVE_DIR))[-1]

    params_path = os.path.join(load_path, [f for f in os.listdir(load_path) if f.startswith("params") and f.endswith(".json")][0])
    model_path = os.path.join(load_path, [f for f in os.listdir(load_path) if f.startswith("model") and f.endswith(".pth")][0])

    with open(params_path, 'r') as f:   
        cfg = json.load(f)

    return cfg


def load_model(cfg = copy.deepcopy(params.params)):

    # p_Y
    pY_cfg = cfg.get("outcome", {})
    pY_name  = pY_cfg.pop("name", None)

    # data.py
    data_cfg = cfg.get("data", {})

    # model.py
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.pop("name", None)
    try:    model_cl = getattr(model, model_name)
    except AttributeError: raise RuntimeError(f"No {model_name} in model.py")
    model_main = model_cl(**model_cfg, **pY_cfg).to(DEVICE) # get model


    data_func = data.generate_data(**data_cfg, **pY_cfg)
    train_cfg = cfg.get("train", {})

    return data_func, model_main, train_cfg
