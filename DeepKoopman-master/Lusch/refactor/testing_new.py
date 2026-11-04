# Serves as a pytorch reimplementation of the training code from the original lusch repo
import os
import pstats
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from time import time
from utils import *
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import helperfns
import importlib
import cProfile
from training_new import Lusch_Model
torch.manual_seed(0)


if __name__ == "__main__":

    exp_name = "Lorenz" # "DiscreteSpectrumExample", "FluidFlowBox", "FluidFlowOnAttractor", "Pendulum", "Lorenz"
    data_subset = {"train":"train1_x", "test":"test_x"}
    wandb_enabled = True 

    koopman_mode = "hot_start" # "koopman" for precomputed koopman, "lusch_params" for randomly-initialized auxillary network, "hot_start" for hot starting auxillary network, "no_hot_start" for using lusch with koop params, but not fillinmg eigenvalues
    params = importlib.import_module("{}ExperimentBestParams".format(exp_name)).params
    helperfns.set_defaults(params)
    learning_rate = params['learning_rate']
    batch_size = 128
    # batch_size = 64
    traj_len = params['len_time']
    k = 10
    # k=params['num_shifts_middle'] # number of middle shifts

    lambdas = {
        "recon": params['recon_lam'],
        "forward": params['recon_lam'],
        "linearity": params['mid_shift_lam'],
        "L1": params["L1_lam"],
        "L2": params["L2_lam"],
        "Linf": params["Linf_lam"]
    }
    
    # Load precomputed koopman operator
    koopman_path = "../../data/koopmans/{}_{}.mat".format(exp_name, "observables")
    koopman, d, omegas = load_koopman(koopman_path)
    koopman = koopman.to(device)
    # Load datasets
    train_dataset = pd.read_csv('../../data/csv/{}_{}.csv'.format(exp_name, data_subset["train"]), delimiter=',', header=None, dtype=np.float64).values
    train_dataset = KoopData(train_dataset, traj_len=traj_len, k=k, d=d)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    test_dataset = pd.read_csv('../../data/csv/{}_{}.csv'.format(exp_name, data_subset['test']), delimiter=',', header=None, dtype=np.float64).values
    test_dataset = KoopData(test_dataset, traj_len=traj_len, k=k, d=d)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # create model
    model = Lusch_Model(params, koopman_mode, koopman=koopman, omegas=omegas, d=d).to(device)
    # override with loaded model


    if koopman_mode == "lusch_params":
        import tensorflow as tf
        tf.compat.v1.disable_eager_execution() 
        # load tf model
        # model_path = "exp1_best_disc/DiscreteSpectrumExample_2025_03_16_01_37_07_841514_model.ckpt"
        # model_path = "exp2_best_pend/Pendulum_2025_03_16_01_37_08_818713_model.ckpt"
        model_path = "exp3_best_fluid/FluidFlowOnAttractor_2025_03_16_02_04_00_059558_model.ckpt"
        checkpoint_path = f"../models/{koopman_mode}/{model_path}"
        if not os.path.exists(checkpoint_path + ".meta"):
            raise ValueError("Model not found at {}".format(checkpoint_path + ".meta"))
        print("Loading model from {}".format(checkpoint_path))

        with tf.compat.v1.Session() as sess:
            # Load the pkl file to get some params
            # pkl_path = checkpoint_path.replace(".ckpt", ".pkl")
            # if not os.path.exists(pkl_path):
            #     raise ValueError("Pkl file not found at {}".format(pkl_path))
            
            # with open(pkl_path, 'rb') as f:
            #     params = pd.read_pickle(f)
            #     print("Loaded params from {}".format(pkl_path))

            # print("params: ", params)

            # Load meta graph
            saver = tf.compat.v1.train.import_meta_graph(checkpoint_path + ".meta")


            saver.restore(sess, checkpoint_path)
            # print("loaded variables: ", tf.compat.v1.global_variables())

            # List all variables
            all_vars = tf.compat.v1.global_variables()
            # print("Loaded variables:", [v.name for v in all_vars])

            saver.restore(sess, checkpoint_path)  # Restore the model
        
            # Get all the variables (parameters)
            all_vars = tf.compat.v1.global_variables()
            # for name, param in model.encoder.state_dict().items():
            #     print(name)            
            # Print the names and values of all variables
            encoder_vars = [var for var in all_vars if 'E' in var.name]
            decoder_vars = [var for var in all_vars if 'D' in var.name]
            # for aux vars, need one list per mini aux model where theyre called OR{i} for i in range(num_aux_models)
            aux_vars= []
            for i in range(len(model.auxillary)):
                aux_vars.append([var for var in all_vars if f'OR{i}' in var.name])

            # Load the encoder and decoder weights into the PyTorch model
            for i in range(len(encoder_vars)):
                if i > len(model.encoder.state_dict())-1:
                    break
                if i % 2 != 0:
                    continue

                    
                # print the name of the variable in encoder, then the name ogf the variable in the tf model
                # print(f"encoder: {i}.weight", encoder_vars[i].name)
                # print(f"encoder: {i}.bias", encoder_vars[i+1].name)
                model.encoder.state_dict()[f"{i}.weight"].copy_(torch.from_numpy(encoder_vars[i].eval().reshape(model.encoder.state_dict()[f"{i}.weight"].shape)))
                model.encoder.state_dict()[f"{i}.bias"].copy_(torch.from_numpy(encoder_vars[i+1].eval().reshape(model.encoder.state_dict()[f"{i}.bias"].shape)))
            
            for i in range(len(decoder_vars)):

                if i > len(model.decoder.state_dict())-1:
                    break
                if i % 2 != 0:
                    continue
                model.decoder.state_dict()[f"{i}.weight"].copy_(torch.from_numpy(decoder_vars[i].eval().reshape(model.decoder.state_dict()[f"{i}.weight"].shape)))
                model.decoder.state_dict()[f"{i}.bias"].copy_(torch.from_numpy(decoder_vars[i+1].eval().reshape(model.decoder.state_dict()[f"{i}.bias"].shape)))
            
            for j, aux_model in enumerate(model.auxillary):
                for i in range(len(aux_vars[j])):
                    if i > len(aux_model.state_dict())-1:
                        break
                    if i % 2 != 0:
                        continue

                    # print(aux_vars[j])
                    # # print the name of the variable in encoder, then the name ogf the variable in the tf model
                    # print(f"aux: {i}.weight", aux_vars[j][i].name)
                    # print(f"aux: {i}.bias", aux_vars[j][i+1].name)
                    aux_model.state_dict()[f"{i}.weight"].copy_(torch.from_numpy(aux_vars[j][i].eval().reshape(aux_model.state_dict()[f"{i}.weight"].shape)))
                    aux_model.state_dict()[f"{i}.bias"].copy_(torch.from_numpy(aux_vars[j][i+1].eval().reshape(aux_model.state_dict()[f"{i}.bias"].shape)))
    
    if koopman_mode == "hot_start":
        model = Lusch_Model(params, "lusch_params", koopman=koopman, omegas=omegas, d=d).to(device)
        # num = "999(1)"
        # ckpt = f"../models/{koopman_mode}/{exp_name}_epoch_{num}.pth"
        ckpt = f"../models/Lorenz_lusch_paramsfresh-valley-115_nc3_nr4_mem1_epoch_499.pth"
        checkpoint = torch.load(ckpt, weights_only=False)

        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        for i, aux_model in enumerate(model.auxillary):
            aux_model.load_state_dict(checkpoint['auxillary_state_dict'][i])
        model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
    model.encoder.eval()
    model.decoder.eval()
    for i, aux_model in enumerate(model.auxillary):
        aux_model.eval()

    n_step_mse_list = []
    full_mse_list = []
    for j, loader in enumerate([train_loader, test_loader]):
        mse_total = 0
        for i, data in enumerate(tqdm(loader, desc="MSE for {}".format(j))):
                og_traj = data[:,:,:,0]
                pred_traj = n_step_predict_vec(model, params, data, n_steps=5,use_log=True)
                f_pred_traj = full_predict_vec(model, params, data, use_log=True)

                # mse between og_traj and pred_traj
                n_step_mse = nn.MSELoss()(pred_traj, og_traj)
                n_step_mse_total += n_step_mse.item()

                full_mse = nn.MSELoss()(f_pred_traj, og_traj)
                mse_total += full_mse.item()
        n_step_mse_list.append(n_step_mse_total / len(loader))
        full_mse_list.append(mse_total / len(loader))

    print("N-step MSE for train: {}, test: {}".format(n_step_mse_list[0], n_step_mse_list[1]))
    print("Full MSE for train: {}, test: {}".format(full_mse_list[0], full_mse_list[1]))

            