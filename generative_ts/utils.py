from . import params, data, model
from .params import DEVICE, BASE_DIR, SAVE_DIR

import os
import json
import copy

from itertools import product

from typing import Optional, Tuple, Dict, Any, List

import torch.nn as nn
import torch

from omegaconf import OmegaConf
from .model import LS4

class Experiment:
    __slots__ = ("data_main", "model_main", "cfg", "save_name")

    def __init__(
        self,
        data_main: nn.Module,
        model_main: nn.Module,
        cfg: Dict[str, Any],
        save_name: str
    ):
        self.data_main = data_main
        self.model_main = model_main
        self.cfg = cfg
        self.save_name = save_name

class Experiment_p(Experiment):
    __slots__ = Experiment.__slots__ + ("load_path",)

    def __init__(
        self,
        data_main: nn.Module,
        model_main: nn.Module,
        cfg: Dict[str, Any],
        save_name: str,
        load_path: str
    ):
        super().__init__(data_main, model_main, cfg, save_name)
        self.load_path = load_path



def load_param(cfg: Dict[str, Any]):
    # 1) outcome 설정
    outcome_cfg = cfg.get("outcome", {})

    # 2) 데이터 모듈 생성
    data_cfg = cfg.get("data", {}).copy()
    data_name = data_cfg.pop("name", None)
    if not hasattr(data, data_name):
        raise RuntimeError(f"No data class '{data_name}' in data.py")
    data_main = getattr(data, data_name)(**data_cfg, **outcome_cfg)

    # 3) 모델 모듈 생성
    model_cfg = cfg.get("model", {}).copy()
    model_name = model_cfg.pop("name", None)
    if model_name == "LS4":
        cfg_path = model_cfg.pop("config_path")
        # 절대 경로 아니면 BASE_DIR 상위로부터 상대경로로 해석
        if not os.path.isabs(cfg_path):
            cfg_path = os.path.normpath(os.path.join(BASE_DIR, os.pardir, cfg_path))
        ls4_conf = OmegaConf.load(cfg_path)
        # 반드시 model block만 넘기기!
        model_main = LS4(ls4_conf.model).to(DEVICE)
    else:
        if not hasattr(model, model_name):
            raise RuntimeError(f"No model class '{model_name}' in model.py")
        model_main = getattr(model, model_name)(**model_cfg, **outcome_cfg).to(DEVICE)

    # 4) save_name 생성
    parts = []
    for section in ("model", "data", "train", "outcome"):
        for k, v in cfg.get(section, {}).items():
            if k in params.save and not (k == "name" and v is None):
                parts.append(f"{k}_{v}")
    save_name = f"{model_name}_" + "_".join(parts)

    return data_main, model_main, cfg, save_name

def load_params(cfg_list: Dict[str, Any] = None) -> List:
    import copy
    from itertools import product

    # 기본 파라미터 복사
    base_cfg = copy.deepcopy(cfg_list or params.params)

    # 리스트형 파라미터 키 모으기
    keys = [
        (sec, k)
        for sec in ("data", "model", "outcome", "train")
        for k, v in base_cfg.get(sec, {}).items()
        if isinstance(v, list)
    ]

    all_cfgs = []
    for vals in product(*(base_cfg[sec][k] for sec, k in keys)):
        new_cfg = copy.deepcopy(base_cfg)
        for (sec, k), v in zip(keys, vals):
            new_cfg[sec][k] = v
        all_cfgs.append(new_cfg)

    # 각 cfg마다 Experiment 생성
    experiments = []
    for cfg in all_cfgs:
        d, m, c, s = load_param(cfg)
        experiments.append(Experiment(d, m, c, s))
    return experiments
