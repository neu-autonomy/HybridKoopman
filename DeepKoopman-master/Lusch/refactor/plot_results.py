# Serves as a pytorch reimplementation of the training code from the original lusch repo
import torch
import numpy as np
import wandb
from time import time
from utils import *
from torch.utils.data import DataLoader
import pandas as pd
import helperfns
torch.manual_seed(0)
import plotly
import importlib
import cProfile
from training_new import Lusch_Model

pr = cProfile.Profile()
pr.enable()

# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
device = torch.device("cpu")

if __name__ == "__main__":
    # Define hyperparams
    # exp_name = "DiscreteSpectrumExample" # "DiscreteSpectrumExample", "FluidFlowBox", "FluidFlowOnAttractor", "Pendulum", "Lorenz"
    exp_name = "Lorenz"
    data_subset = {"train":"train1_x", "test":"test_x"}
    # wandb_enabled =False 
    ckpt = f"../models/Lorenz_lusch_paramsrevived-feather-129_nc2_nr2_mem1_epoch_600.pth"
    wandb_enabled =False
    use_aux = True
    # koopman_mode = "hot_start"
    koopman_mode = "lusch_params"
    params = importlib.import_module("{}ExperimentBestParams".format(exp_name)).params
    helperfns.set_defaults(params)
    learning_rate = params['learning_rate']
    batch_size = 25
    traj_len = params['len_time']
    params['num_shifts_middle'] = 3
    k = params['num_shifts_middle']
    # traj_len = params['num_shifts']
    # k = params['num_shifts_middle']
    lambdas = {
        "recon": params['recon_lam'],
        "forward": params['recon_lam'],
        "linearity": params['mid_shift_lam'],
        "L1": params["L1_lam"],
        "L2": params["L2_lam"],
        "Linf": params["Linf_lam"]
    }

    # # Load precomputed koopman operatoTr
    koopman_path = "../../data/koopmans/{}_{}.mat".format(exp_name, "observables") # TODO remove fake
    koopman, d, omegas  = load_koopman(koopman_path)
    koopman = koopman.to(device)

    # d = 2
    # omegas = [0.0, 0.0, 1j, 1j, 1j, 1j]
    # koopman = torch.eye(6)
    
    hidden_size = koopman.shape[0]
    train_dataset = pd.read_csv('../../data/csv/{}_{}.csv'.format(exp_name, data_subset["train"]), delimiter=',', header=None, dtype=np.float64).values

    train_min = np.min(train_dataset, axis=0)
    train_max = np.max(train_dataset, axis=0)
    
    val_dataset = pd.read_csv('../../data/csv/{}_{}.csv'.format(exp_name, data_subset["test"]), delimiter=',', header=None, dtype=np.float64).values
    # val_dataset = 2.0 * (val_dataset - train_min) / (train_max - train_min) - 1.0
    val_dataset = KoopData(val_dataset, traj_len=traj_len, k=4, d=2)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    # Define model
    params['num_complex_pairs'] = 2
    params['num_real'] = 2
    params['wandb_enabled'] = False

    model = Lusch_Model(params, koopman_mode=koopman_mode, koopman=koopman, omegas=omegas, d=d).to(device)
    d = 1
    plot_lusch(model, 
            params=params,
            val_loader=val_loader, 
            load=ckpt, 
            name=exp_name + "-aux-" + str(use_aux) + "-koopman_mode-" + str(koopman_mode) + "1")
    
    