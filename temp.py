from generative_ts import params, data, model, utils, test, train
from generative_ts.params import DEVICE
from generative_ts.utils import Experiment, Experiment_p

import os
import numpy as np
import matplotlib.pyplot as plt 

'''
test.plot_posterior(utils.load_model('250709_083749'), t_0_ratio=2)


from multiprocessing import Pool

def run(expr):
    train(expr, verbose=1)

if __name__ == "__main__":
    exprs = utils.load_params()   
    with Pool(processes=30) as pool:
        pool.map(run, exprs)
'''

for expr in utils.load_params():
    train.train(expr)
