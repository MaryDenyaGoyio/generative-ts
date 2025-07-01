from . import params, data, model
from .params import DEVICE, BASE_DIR, SAVE_DIR

import os
import json
import copy

import torch

def load_model(target = None):

    load_path = path if (target is not None) and os.path.exists(path:=os.path.join(SAVE_DIR, target)) else sorted([ dir for name in os.listdir(SAVE_DIR) if os.path.isdir(dir := os.path.join(SAVE_DIR, name)) ])[-1]

    params_path = os.path.join(load_path, [f for f in os.listdir(load_path) if f.startswith("params") and f.endswith(".json")][0])
    model_path = os.path.join(load_path, [f for f in os.listdir(load_path) if f.startswith("model") and f.endswith(".pth")][0])

    with open(params_path, 'r') as f:   
        cfg = json.load(f)


    data_func, model_main, _, save_name = load_params(cfg)

    model_main.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model_main.eval()

    return data_func, model_main, save_name, load_path



def load_params(cfg = copy.deepcopy(params.params)):

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


    def data_func():    return data.generate_data(**data_cfg, **pY_cfg)
    train_cfg = cfg.get("train", {})

    save_name = f"{model_name}_" + f"_".join(
            f"{k}_{v}"
            for section in ('model', 'data', 'train', 'outcome')
            for k, v in cfg[section].items()
            if (k in params.save) and not (k == 'name' and v is None)
        )

    return data_func, model_main, train_cfg, save_name
