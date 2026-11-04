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
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + f"({counter})" + extension
        counter += 1
    return path

class Lusch_Model(nn.Module):
    def __init__(self, params, koopman_mode, koopman=None, omegas=None, d=1):
        super(Lusch_Model, self).__init__()
        self.koopman = koopman.to(device)
        
        self.koop_complex_omegas = []
        self.koop_real_omegas = []
        for omega in omegas:
            # if the omega is complex
            if omega.imag != 0:
                self.koop_complex_omegas.append(omega)
            else:
                self.koop_real_omegas.append(omega.real)

        # print("Koopman shape: ", self.koopman.shape)
        depth = int((params['d'] - 4) / 2)

        widths = params['widths']
        encoder_widths = widths[0:depth + 2]
        decoder_widths = widths[depth + 2 :]

        if koopman_mode == "lusch_params":
            encoder_widths[-1] = params['num_real']+params['num_complex_pairs']*2
            decoder_widths[0] = params['num_real']+params['num_complex_pairs']*2
            params['memory'] = 1

        else:
            
            encoder_widths[0]  =  int(encoder_widths[0] * d)
            decoder_widths[-1] = int(decoder_widths[-1] * d)
            encoder_widths[-1] = int(koopman.shape[0])
            decoder_widths[0]  = int(koopman.shape[0])
            params['num_complex_pairs'] = int(len(self.koop_complex_omegas)/2)
            params['num_real'] = len(self.koop_real_omegas)
            params['memory'] = d

        print(f'en: {encoder_widths[0]}')
        print(f'de: {decoder_widths[-1]}')

        # Define encoder
        encoder_layers = []
        for i in range(encoder_widths.__len__() - 1):
            encoder_layers.append(nn.Linear(encoder_widths[i], encoder_widths[i + 1]))
            encoder_layers.append(nn.ReLU())
        encoder_layers.pop()
        self.encoder = nn.Sequential(*encoder_layers)
        self.encoder.to(device)

        # Define decoder
        decoder_layers = []
        for i in range(decoder_widths.__len__() - 1):
            decoder_layers.append(nn.Linear(decoder_widths[i], decoder_widths[i + 1]))
            decoder_layers.append(nn.ReLU())
        decoder_layers.pop()
        self.decoder = nn.Sequential(*decoder_layers)
        self.decoder.to(device)

        auxillary_widths_complex = params['widths_omega_complex']
        auxillary_widths_real = params['widths_omega_real']
        
        # create one aux network per complex pair and real pair
        num_complex_pairs = params['num_complex_pairs']
        num_real_pairs = params['num_real']
        print("N complex and real pairs: ", num_complex_pairs, num_real_pairs)

        self.auxillary = []
        for i in range(num_complex_pairs):
            auxillary_layers = []
            for j in range(auxillary_widths_complex.__len__() - 1):
                auxillary_layers.append(nn.Linear(auxillary_widths_complex[j], auxillary_widths_complex[j + 1]))
                auxillary_layers.append(nn.ReLU())
            auxillary_layers.pop()
            self.auxillary.append(nn.Sequential(*auxillary_layers))
            self.auxillary[i].to(device)
        
        for i in range(num_real_pairs):
            auxillary_layers = []
            for j in range(auxillary_widths_real.__len__() - 1):
                auxillary_layers.append(nn.Linear(auxillary_widths_real[j], auxillary_widths_real[j + 1]))
                auxillary_layers.append(nn.ReLU())
            auxillary_layers.pop()
            self.auxillary.append(nn.Sequential(*auxillary_layers))
            self.auxillary[i+num_complex_pairs].to(device)

        # self.display_networks(params)


    def display_networks(self, params):
        # use torchvision to display the network with dummy data
        nets = {"Encoder":self.encoder, "Decoder":self.decoder}
        # add aux to nets
        aux_list = [("Auxillary{}".format(i), self.auxillary[i]) for i in range(len(self.auxillary))]
        nets.update(aux_list)
        print(nets)
        for name, net in nets.items():
            net_input = net[0].weight.shape[1]
            dummy_data = torch.randn(1, net_input).to(device)
            y = net(dummy_data)
            # uncomment to save nets to a file as pngs
            # make_dot(y, params=dict(net.named_parameters()), show_attrs=True, show_saved=True).render(name + ".png")
        

    def forward(self, data):
        batched_x_t = data.reshape(-1, data.shape[2], data.shape[3])  # Shape: (batches * num_timesteps, num_components, k)
        x = self.encoder(batched_x_t[:,:,0])  # Shape: (batches * num_timesteps, num_components, out_dim)
        x = self.decoder(x)

        # unbatch
        x = x.reshape(data.shape[0], data.shape[1], data.shape[2])
        return x

def reconstruct_loss(epoch, params, model, x_t, k, d, lambdas, koopman_mode):
    # k is number of middle shifts
    # d is memory
    wandb_enabled=params['wandb_enabled']

    start_time = time()
    encoder = model.encoder.to(device)
    decoder = model.decoder.to(device)
    auxillary = model.auxillary
    koop = model.koopman
    x_t = x_t.to(device) # batches, numtimesteps, num_components, k 
    batches,num_timesteps, num_components, _k = x_t.shape
    num_complex_pairs = params['num_complex_pairs']
    num_real = params['num_real']
    num_obsv = int((2*num_complex_pairs+num_real)/d)
    # reshape the data into trajectories
    batched_x_t = x_t.reshape(-1, x_t.shape[2], x_t.shape[3])  # Shape: (batches * num_timesteps, num_components, k)

    # extract reference states
    x_tn = batched_x_t[:, :, 0:d].permute(0,2,1) # Shape: (batches * num_timesteps, d, num_components)
    # x_tk = batched_x_t[:, :, 1:].permute(2,0,1) # Shape: (k-1, batches * num_timesteps, num_components)
    x_tk = batched_x_t[:, :, 1:].permute(0,2,1) # Shape: (batches * num_timesteps, k-1, num_components) ##################
    # x_tk = torch.stack([batched_x_t[:,:, i:i+d] for i in range(k-d+1)], dim=-1).permute(0, 2, 3, 1) # shape: (batches * num_timesteps,d, k-d+1, num_components)
    encoded_x_tn = encoder(x_tn.reshape(-1, x_tn.shape[-1])) # Shape: (batches * num_timesteps*d, num_obsv)
    
    decoded_encoded_x_tn = decoder(encoded_x_tn) # Shape: (batches * num_timesteps*d, num_components) 
    reconstruction_loss = nn.MSELoss(reduction="none")(decoded_encoded_x_tn.reshape(*x_tn.shape), x_tn)
    reconstruction_loss = torch.sum(lambdas['recon'] * reconstruction_loss)

    if wandb_enabled:
        wandb.log({"Epoch":epoch+1, "Solo Reconstruction Loss": reconstruction_loss})

    return reconstruction_loss

# @torch.compile
def vec_lusch_loss(epoch, params, model, x_t, k, d, lambdas, koopman_mode, use_log_real=False):

    encoder = model.encoder.to(device)
    decoder = model.decoder.to(device)
    auxillary = model.auxillary
    koop = model.koopman
    x_t = x_t.to(device) # batches, numtimesteps, num_components, k 
    batches,num_timesteps, num_components, _k = x_t.shape
    num_complex_pairs = params['num_complex_pairs']
    num_real = params['num_real']
    num_obsv = int((2*num_complex_pairs+num_real)/d)
    # reshape the data into trajectories
    batched_x_t = x_t.reshape(-1, x_t.shape[2], x_t.shape[3])  # Shape: (batches * num_timesteps, num_components, k)

    # extract reference states
    x_tn = batched_x_t[:, :, 0:d].permute(0,2,1) # Shape: (batches * num_timesteps, d, num_components)
    # x_tk = batched_x_t[:, :, 1:].permute(2,0,1) # Shape: (k-1, batches * num_timesteps, num_components)
    x_tk = batched_x_t[:, :, 1:].permute(0,2,1) # Shape: (batches * num_timesteps, k-1, num_components) ##################
    # x_tk = torch.stack([batched_x_t[:,:, i:i+d] for i in range(k-d+1)], dim=-1).permute(0, 2, 3, 1) # shape: (batches * num_timesteps,d, k-d+1, num_components)
    encoded_x_tn = encoder(x_tn.reshape(-1, x_tn.shape[-1])) # Shape: (batches * num_timesteps*d, num_obsv)

    if koopman_mode == "koopman":
        encoded_x_tn = encoded_x_tn.reshape(x_tn.shape[0], d, encoded_x_tn.shape[-1]) # Shape: (batches * num_timesteps, num_components, k)
        koop_stacked = torch.stack([torch.linalg.matrix_power(koop, k) for k in range(0, k)], dim=0).to(device) # Shape: (k, num_components, num_components)
        koop_encoded_x_tn = (koop_stacked @ encoded_x_tn.squeeze(1).permute(1,0)).permute(0,2,1) # Shape: (batches * num_timesteps, num_components, k)
        gt_x_tk = x_tk
    else:
        gt_x_tk =  torch.zeros(((k-d+1), x_tn.shape[0] * x_tn.shape[1], x_tn.shape[2])).to(device)
        # gt_encoded_x_tk =  torch.zeros((k-d+1, encoded_x_tn.shape[0], encoded_x_tn.shape[1])).to(device)
        koop_encoded_x_tn = torch.zeros((k-d+1, batches*num_timesteps, d*num_obsv)).to(device) ##########
        koop_encoded_x_tn_i = encoded_x_tn.reshape(batches*num_timesteps,d*num_obsv) # Shape: (batches * num_timesteps, out_dim*d)

        for i in range(0, k-d+1): ##############  either k-d or k or k-d+1

            nn_omegas_complex, nn_omegas_real = get_omegas_from_auxillary(model, params, koop_encoded_x_tn_i)

            if use_log_real:
                koopman = form_pkoop_from_auxillary(params, nn_omegas_complex, nn_omegas_real, delta_t=1.0).to(device) 
            else:
                koopman = form_koopman_from_auxillary(params, nn_omegas_complex, nn_omegas_real, delta_t=1.0)

            if torch.any(torch.isnan(koopman)) or torch.any(torch.isinf(koopman)):
                # return 
                breakpoint()
                print("uh oh")

            # apply koopman to encoded state
            koop_encoded_x_tn_i = torch.bmm(koopman, koop_encoded_x_tn_i.unsqueeze(-1)).squeeze(-1)
            koop_encoded_x_tn[i, :, :] = koop_encoded_x_tn_i.unsqueeze(0) # Shape: ((1, batches*num_timesteps, out_dim*d)
            gt_x_tk[i,:,:] = batched_x_t[:, :, i+1:i+d+1].permute(0,2,1).reshape(-1, x_tn.shape[-1])

            if torch.any(torch.isnan(koop_encoded_x_tn_i)) or torch.any(torch.isinf(koop_encoded_x_tn_i)):
                # return
                breakpoint()
                print("uh oh")

    gt_encoded_x_tk = encoder(gt_x_tk.reshape(-1, x_tk.shape[-1]))
    decoded_encoded_x_tn = decoder(encoded_x_tn) # Shape: (batches * num_timesteps*d, num_components) 

    # apply the decoder to the koopman encoded states - cant do all ks at once, so flatten the k dimension
    # koop_encoded_x_tn.reshape(-1, koop_encoded_x_tn.shape[-1]) -> Shape: ((k-d+1)*batches*num_timesteps*d, out_dim)
    decoded_koop_encoded_x_tk = decoder(koop_encoded_x_tn.reshape(-1, koop_encoded_x_tn.shape[-1])) # Shape: ((k-d+1)*batches * num_timesteps*d, num_components)

    reconstruction_loss = nn.MSELoss(reduction="none")(decoded_encoded_x_tn.reshape(*x_tn.shape), x_tn)
    linearity_loss = nn.MSELoss(reduction='none')(gt_encoded_x_tk, koop_encoded_x_tn.reshape(-1, num_obsv))
    forward_loss = nn.MSELoss(reduction='none')(decoded_koop_encoded_x_tk, gt_x_tk.reshape(-1, num_components))

    # lambda2 = torch.pow(lambdas['linearity'], torch.arange(1, k+1)).to(device)
    lambda2 = torch.pow(lambdas['linearity'], torch.arange(1, k+1))
    lambda2 = torch.kron(torch.ones(1,batches*num_timesteps*d), lambda2).squeeze().to(device)
    
    lambda3 = torch.pow(lambdas['forward'], torch.arange(1, k+1))
    lambda3 = torch.kron(torch.ones(1,batches*num_timesteps*d), lambda3).squeeze().to(device)
    # mean across all dims but given dim
    reconstruction_loss = torch.mean(reconstruction_loss)
    linearity_loss = torch.mean(linearity_loss, dim = [1]) ###### dim=[1,2]
    forward_loss = torch.mean(forward_loss, dim = [1])###### dim=[1,2]

    epsilon = 1e-8
    Linf1 = torch.norm(decoded_encoded_x_tn - x_tn, p=float("inf"), dim=[1,2])
    Linf2 = torch.norm(decoded_koop_encoded_x_tk.reshape(*gt_x_tk.shape) - gt_x_tk, p=float('inf'), dim=[1,2])  

    Linf1_den = torch.norm(x_tn, p=float('inf'), dim=[1,2]) + epsilon
    Linf2_den = torch.norm(gt_x_tk, p=float('inf'), dim=[1,2]) + epsilon

    Linf1_penalty = torch.sum(Linf1 / Linf1_den)
    Linf2_penalty = torch.sum(Linf2 / Linf2_den)

    loss_Linf = lambdas['Linf'] * (Linf1_penalty + Linf2_penalty)
    # loss_Linf = 0.0

    reconstruction_loss = torch.sum(lambdas['recon'] * reconstruction_loss)
    # dot product of the linearity loss and the lambda2 vector
    linearity_loss = torch.sum(torch.dot(linearity_loss, lambda2)) # TODO This sum is redundant. Dot already makes it 1D
    forward_loss = torch.sum(torch.dot(forward_loss, lambda3))

    full_loss = reconstruction_loss + linearity_loss + forward_loss + loss_Linf
    # print("Reconstruction, Forward, Linearity Loss: ", reconstruction_loss, forward_loss, linearity_loss)
    # print(f"Time taken to compute loss: {end_time - start_time} seconds")

    # if koopman_mode != "koopman": #If we are using the auxillary network, we need to return the koopman operator for visualization
    #     cache = {
    #         "koopman": koopman,
    #     }
    # else:
    cache = None

    return full_loss, (reconstruction_loss, forward_loss, linearity_loss, loss_Linf), cache


def add_regularization(model, rec_loss, loss, lambdas):
    if lambdas['L1'] == 0.0 and lambdas['L2'] == 0.0:
        return rec_loss, loss
    # for each parameter in the mkodel, add penalty for high weights
    loss_L1 = 0
    loss_L2 = 0
    # for param in model.parameters():
        # if len(param.shape) > 1: # dont regularize biases
            # loss_L1 += lambdas["L1"] * torch.norm(param, p=1)
            # loss_L2 += lambdas["L2"] * torch.norm(param, p=2)**2

    regularized_rec_loss = rec_loss + loss_L1 + loss_L2
    regularized_loss = loss + loss_L1 + loss_L2

    return regularized_rec_loss, regularized_loss

def pretrain_aux(model, params, train_loader, learning_rate,  d=1):
    # Collect all parameters from auxiliary networks
    auxillary = model.auxillary
    encoder = model.encoder
    wandb_enabled = params['wandb_enabled']

    all_params = sum([list(n.parameters()) for n in auxillary], [])
    aux_optimizer = optim.Adam(all_params, lr=learning_rate, weight_decay=1e-13)
    gt_koop_complex_omegas = model.koop_complex_omegas
    gt_koop_real_omegas = model.koop_real_omegas
    
    num_complex_pairs = int(len(gt_koop_complex_omegas)/2)
    num_real_pairs = len(gt_koop_real_omegas)

    pretrain_epochs = 200
    for epoch in tqdm(range(pretrain_epochs), desc="Pretraining Epochs", colour="red"):
        loss_total = 0
        for i, batch in enumerate(train_loader):
            # Zero the gradients
            aux_optimizer.zero_grad()
            batches, traj_len, dim, k = batch.shape
            batched_x_t = batch.reshape(-1, batch.shape[2], batch.shape[3])  # (batches*traj_len, dim, k)
            x_tn = batched_x_t[:, :, 0:d] #  (batches*traj_len, dim, d)
            x_tn = x_tn.permute(0,2,1) # (batches*traj_len, d, dim)
            x_tn = x_tn.reshape(batches*traj_len*d, dim) # (batches*traj_len* d, dim)

            encoded_x_tn = encoder(x_tn.to(device))   #( batches*traj_len*d, num_obsv)
            encoded_x_tn = encoded_x_tn.reshape(batches*traj_len,-1)   #( batches*traj_len, d*num_obsv)

            nn_omegas_complex, nn_omegas_real = get_omegas_from_auxillary(model, params, encoded_x_tn)

            if len(nn_omegas_complex) != 0:
               
                nn_omegas_complex = torch.stack(nn_omegas_complex, dim=1) 
                gt_koop_complex_omegas = torch.tensor(gt_koop_complex_omegas, device=device)
                # gt_complex_re = torch.tensor(gt_koop_complex_omegas.real, dtype=torch.float32)
                gt_complex_re = gt_koop_complex_omegas.real.clone().detach().type(torch.float32)
                # gt_complex_imag = torch.tensor(gt_koop_complex_omegas.imag, dtype=torch.float32)
                gt_complex_imag = gt_koop_complex_omegas.imag.clone().detach().type(torch.float32)
                # reals = []
                # imags = []
                # for (re, img) in zip(gt_complex_re, gt_complex_imag):
                #     if re in reals and img in imags:
                #         continue
                #     else:
                #         reals.append(re)
                #         imags.append(img)
                gt_complex_re = gt_complex_re[torch.arange(0, num_complex_pairs*2,2)]                
                gt_complex_imag = gt_complex_imag[torch.arange(0, num_complex_pairs*2,2)]                

                koop_complex_omegas = torch.hstack((gt_complex_re, gt_complex_imag)) 

                koop_complex_omegas = koop_complex_omegas.repeat(nn_omegas_complex.shape[0], 1, 1)


                nn_omegas_complex = nn_omegas_complex.reshape(*koop_complex_omegas.shape)

            
            if len(nn_omegas_real) != 0:
                
                nn_omegas_real = torch.stack(nn_omegas_real, dim=1)

                koop_real_omegas = torch.tensor(np.array(gt_koop_real_omegas), device=device, dtype=torch.float32)

                # duplicate koop omegas to match the size of the nn omegas
                koop_real_omegas = koop_real_omegas.repeat(nn_omegas_real.shape[0], 1)

                nn_omegas_real = nn_omegas_real.reshape(*koop_real_omegas.shape)

            # Compute loss
            if len(nn_omegas_complex) > 0:
                if len(nn_omegas_real) > 0:
                    loss = nn.MSELoss()(nn_omegas_complex, koop_complex_omegas) + nn.MSELoss()(nn_omegas_real, koop_real_omegas)
                else:
                    loss = nn.MSELoss()(nn_omegas_complex, koop_complex_omegas)
            elif len(nn_omegas_real) > 0:
                loss = nn.MSELoss()(nn_omegas_real, koop_real_omegas)
            else:
                loss = 0.0

            loss_total += loss

            # Backward pass and optimization
            loss.backward()
            aux_optimizer.step()
            if wandb_enabled:
                wandb.log({"Auxiliary Pretraining Loss": loss.item()})
    return loss

def train_lusch(model, params, lambdas, train_loader, val_loader, test_loader, k, d, learning_rate, koopman_mode,  load=None, name=None, use_log=False, start_epoch=0):

    if koopman_mode is None:
        raise ValueError("Please specify whether to use an auxillary network or not")

    # Define the model
    encoder = model.encoder
    decoder = model.decoder
    auxillary = model.auxillary
    koopman = model.koopman
    wandb_enabled = params['wandb_enabled']
    exp_name = params['exp_name']

    nc = params['num_complex_pairs'] 
    nr = params['num_real']
    mem = params['memory']
    if koopman_mode == "koopman":
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate, weight_decay=1e-13)
    else:
        all_params = sum([list(n.parameters()) for n in auxillary], [])
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()) + all_params, lr=learning_rate, weight_decay=1e-13)

    # Define hyperparams and tracking variables
    epochs = 1_000
    # epochs = 300
    recon_epochs = 100
    freeze_epochs = 100
    start_epoch = params['start_epoch']
    swap_mode = params['swap_mode']
    loss_over_time = []
    val_loss_over_time = []

    start_time = time()

    # load the model if applicable
    if load is not None:
        # load the model
        checkpoint = torch.load(load)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        for net, net_state in zip(auxillary, checkpoint['auxillary_state_dict']):
            net.load_state_dict(net_state)
        # auxillary.load_state_dict(checkpoint['auxillary_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        start_time = checkpoint['start_time']
        print("Loaded model from ", load)

    log_mse_n = 10
    test_mse = np.zeros((int(np.ceil(epochs/log_mse_n)), 3))

    print("Starting at epoch: ", start_epoch)
    for epoch in tqdm(range(start_epoch, epochs), desc="Training", colour="green"):
        # Set the model to training mode
        model.train()

        # Collect losses to get average over epoch
        loss_per_epoch = 0

        # Training loop
        for train_iter, batch in enumerate(train_loader):

            # Zero the gradients
            optimizer.zero_grad()

            # Backward pass and optimization
            if epoch < recon_epochs:
                # print(batch.device)
                loss_rec = reconstruct_loss(epoch, params, model, batch , k, d, lambdas, koopman_mode)
                # loss_rec, _ = add_regularization(model, loss_rec, 0, lambdas)
                loss = loss_rec
                loss_rec.backward()
            else:
                if swap_mode == True:
                    if epoch == recon_epochs and koopman_mode == "hot_start":
                        print(f'recon loss {loss}')
                        p_loss = pretrain_aux(model, params, train_loader,  learning_rate, d)
                        print(f'pretrain loss {p_loss}')
                        for aux in model.auxillary:
                            for param in aux.parameters():
                                param.requires_grad = False
                        break

                    elif epoch - recon_epochs % freeze_epochs == 0:
                        print(f'aux froze at epoch {epoch}')
                        for aux in model.auxillary:
                            for param in aux.parameters():
                                param.requires_grad = False
                        for param in encoder.parameters():
                            param.requires_grad = True
                        for param in decoder.parameters():
                            param.requires_grad = True
                        break

                    elif epoch - recon_epochs % freeze_epochs == freeze_epochs // 2:
                        print(f'enc froze at epoch {epoch}')
                        for aux in model.auxillary:
                            for param in aux.parameters():
                                param.requires_grad = True
                        for param in encoder.parameters():
                            param.requires_grad = False
                        for param in decoder.parameters():
                            param.requires_grad = False
                        break
                else:
                    if epoch == recon_epochs and koopman_mode == "hot_start":
                        pretrain_aux(model, params, train_loader,  learning_rate, d)
                        for aux in model.auxillary:
                            for param in aux.parameters():
                                param.requires_grad = False
                        break

                    elif epoch == recon_epochs + freeze_epochs and koopman_mode == "hot_start":
                        for aux in model.auxillary:
                            for param in aux.parameters():
                                param.requires_grad = True
                        break
                loss, (r_loss, f_loss, l_loss, linf_loss), cache = vec_lusch_loss(epoch, params, model, batch, k, d, lambdas, koopman_mode,use_log_real=use_log)
                # loss_rec, loss = add_regularization(model, r_loss, loss, lambdas)
                loss.backward()

            # temp test: use mse loss for only encode/decode
            # batch = batch.to(device)
            # Recon_loss = nn.MSELoss()(model(batch), batch[:,:,:,0])
            # Recon_loss.backward()

            optimizer.step()

            loss_per_epoch += loss.item()

            # end_time = time()
            # print(f"Time taken to  run one batch {end_time - start_time} seconds")

        loss_over_time.append(loss_per_epoch / len(train_loader))
        
        if wandb_enabled and epoch > recon_epochs:
            wandb.log({"Epoch":epoch+1, "Train Loss": loss_over_time[-1]})

        # Validation error
        if epoch % 20 == 0 and epoch > recon_epochs:
            train_error = loss.item()
            # get val error
            for i, batch in enumerate(val_loader):
                val_loss, (r_loss, f_loss, l_loss, linf_loss), cache = vec_lusch_loss(epoch, params, model, batch, k, d, lambdas, koopman_mode, use_log_real=use_log)
                # val_loss_rec, val_loss = add_regularization(model, r_loss, val_loss, lambdas)

            val_loss = val_loss.item()
            print(f"Epoch: {epoch}, Train Loss: {train_error}, Val Loss: {val_loss}")
            
            val_loss_over_time.append(val_loss)

            if wandb_enabled:
                wandb.log({"Epoch":epoch+1, "Val Loss": val_loss})

        if epoch % log_mse_n == 0 and epoch > recon_epochs:
            total_norm_mse = 0
            total_norm_mse_5 = 0
            total_norm_mse_f = 0
            with torch.no_grad():
                for batch in test_loader:
                    # og_traj = batch[:,0].permute(32\3232\320, 2, 1).to(device)
                    og_traj = test_loader.dataset.reconstruct_one_batch(batch).to(device)
                    # test_batch = batch.to(device)
                    use_full_predict = False

                    if use_full_predict:
                        # pred_traj = full_predict_vec(model, params, og_traj, n_steps=1, use_log=use_log)
                        # pred_traj_5 = full_predict_vec(model, params, og_traj, n_steps=5, use_log=use_log)
                        # pred_traj_f = full_predict_vec(model, params, og_traj, n_steps=params['len_time']-1, use_log=use_log)

                        # # print(pred_traj.shape, og_traj.shape)
                        # loss_fun = nn.MSELoss()
                        # mse = loss_fun(pred_traj, og_traj)
                        # mse_5 = loss_fun(pred_traj_5, og_traj)
                        # mse_f = loss_fun(pred_traj_f, og_traj)
                        raise ValueError('full predict vec was not implemented yet with memory')

                    else:
                        # pred_traj,_ = n_step_predict_vec(model, params, og_traj, n_steps=1, use_log=use_log)
                        # pred_traj_5,_ = n_step_predict_vec(model, params, og_traj, n_steps=5, use_log=use_log)
                        # pred_traj_f,_ = n_step_predict_vec(model, params, og_traj, n_steps=params['len_time']-1, use_log=use_log)
                        _, pred_traj = n_step_predict_vec(model, params, og_traj, n_steps=1, use_log=use_log)
                        # _, pred_traj_5 = n_step_predict_vec(model, params, og_traj, n_steps=5, use_log=use_log)
                        # _, pred_traj_f = n_step_predict_vec(model, params, og_traj, n_steps=params['len_time']-1, use_log=use_log)

                        # print(pred_traj.shape, og_traj.shape)
                        loss_fun = nn.MSELoss()

                        # mse = loss_fun(pred_traj[:,1:,:], og_traj[:,1:,:])
                        # mse_5 = loss_fun(pred_traj_5[:,5:,:], og_traj[:,5:,:])
                        # mse_f = loss_fun(pred_traj_f[-1], og_traj[-1])

                        mse = loss_fun(pred_traj, test_loader.dataset.reconstruct_one_batch_k(batch, 1).to(device))
                        # mse_5 = loss_fun(pred_traj_5, test_loader.dataset.reconstruct_one_batch_k(batch, 5).to(device))
                        # mse_f = loss_fun(pred_traj_f, test_loader.dataset.reconstruct_one_batch_k(batch, params['len_time']-1).to(device))


                    # print("MSE {}, MSE 5 {}, MSE Full {}".format(mse, mse_5, mse_f))

                    total_norm_mse += mse.item()
                    # total_norm_mse_5 += mse_5.item()
                    # total_norm_mse_f += mse_f.item()
            
            total_norm_mse /= len(test_loader)
            total_norm_mse_5 /= len(test_loader)
            total_norm_mse_f /= len(test_loader)

            # print(f"Epoch: {epoch}, Test MSE: {total_norm_mse}, Test MSE 5: {total_norm_mse_5}, Test MSE Full: {total_norm_mse_f}")
            np_path = f"../models/{exp_name}_{name}_nc{nc}_nr{nr}_mem{mem}_loss.txt"
            test_mse[epoch//log_mse_n,0] = total_norm_mse
            # test_mse[epoch//log_mse_n,1] = total_norm_mse_5
            # test_mse[epoch//log_mse_n,2] = total_norm_mse_f

            np.savetxt(np_path, test_mse)

            if wandb_enabled:
                # wandb.log({"Epoch":epoch+1, "Test MSE": total_norm_mse, "Test MSE 5": total_norm_mse_5, "Test MSE Full": total_norm_mse_f})
                wandb.log({"Epoch":epoch+1, "Test MSE": total_norm_mse})

        # Check for convergence to save time
        # if (epoch > 2 and (abs(loss_over_time[-1] - loss_over_time[-2]) < 1e-10) or epoch == epochs-1):

        if epoch % 200 == 0:

            path = uniquify(f"../models/{exp_name}_{name}_nc{nc}_nr{nr}_mem{mem}_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'auxillary_state_dict': [n.state_dict() for n in auxillary],
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'start_time': start_time,
                'exp_params': params
            }, path)

    print("Converged at epoch: ", epoch) 
    converged_time = time()
    print(f"Time taken to converge: {converged_time - start_time} seconds")


    path = uniquify(f"../models/{exp_name}_{name}_nc{nc}_nr{nr}_mem{mem}_epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'auxillary_state_dict': [n.state_dict() for n in auxillary],
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
        'start_time': start_time
    }, path)

    # get final koopman
    dummy_data = torch.randn(*batch.shape).to(device)
    dummy_data = dummy_data.reshape(-1, dummy_data.shape[2], dummy_data.shape[3])  # Shape: (batches * num_timesteps, num_components, k)
    encoder = model.encoder.to(device)
    auxillary = [n.to(device) for n in model.auxillary]
    
    encoded_data = encoder(dummy_data[:,:,0])

    nn_omegas_complex, nn_omegas_real = get_omegas_from_auxillary(model, params, encoded_data)

    # koopman_aux = form_koopman_from_auxillary(params, nn_omegas_complex, nn_omegas_real, delta_t=1.0)
    if use_log:
        koopman_aux = form_pkoop_from_auxillary(params, nn_omegas_complex, nn_omegas_real, delta_t=1.0)
    else:
        koopman_aux = form_koopman_from_auxillary(params, nn_omegas_complex, nn_omegas_real, delta_t=1.0)
    koopman_path = uniquify("../../data/koopmans/{}_{}_nc{}_nr{}_mem{}.txt".format(exp_name, name, nc, nr, mem))

    print("koopman_aux shape: ", koopman_aux.shape)
    # output str koopman to txt file
    koopman_str = koopman_aux.cpu().detach().numpy()[0]
    np.savetxt(koopman_path, koopman_str, delimiter=',')
    print("Saved Koopman to ", koopman_path)

    return path


def run_exp(params, data_subset = {"train":"train1_x", "test":"test_x"}, load=None, start_epoch=0):
    torch.manual_seed(0)
    pr = cProfile.Profile()
    pr.enable()
    learning_rate = params['learning_rate']
    traj_len = params['len_time']
    koopman_mode = params['koopman_mode']
    batch_size = params['batch_size']
    use_log = params['use_log_real']
    load_model = params['load_model']
    k=params['num_shifts_middle'] # number of middle shifts

    lambdas = {
        "recon": params['recon_lam'],
        "forward": params['recon_lam'],
        "linearity": params['mid_shift_lam'],
        "L1": params["L1_lam"],
        "L2": params["L2_lam"],
        "Linf": params["Linf_lam"]
    }
    exp_name = params['exp_name']

    # Load precomputed koopman operator
    if "koopman_path" not in params:
        koopman_path = "../../data/koopmans/{}_{}.mat".format(exp_name, "observables")
    else:
        koopman_path =  params["koopman_path"]
    koopman, memory, omegas = load_koopman(koopman_path, use_log=use_log)
    koopman = koopman.to(device)
    if koopman_mode == "lusch_params": #memory 1 no matter what
        memory = 1


    print("Koopman shape: ", koopman.shape)
    print("Memory/System Order: ", memory)
    
    if params['wandb_enabled']:
        wandb.init(project="KoopmanGrowth", 
                config={"learning_rate": learning_rate, 
                    "batch_size": batch_size, 
                    "traj_len": traj_len, 
                    "k - history": k,
                    "reconstruction loss": lambdas['recon'],
                    "forward loss": lambdas['forward'],
                    "linearity loss": lambdas['linearity'],
                    "koopman_mode": koopman_mode,
                    })
        

        wandb_name = wandb.run.name
        extra_name = "Precomputed Koopman"
        extra_name = extra_name + "-" if extra_name != "" else extra_name
        wandb.run.name = exp_name + "-" + extra_name + wandb.run.name
        params['wandb_name'] = wandb.run.name
    # print(batch_size)
    # Load dataset
    # Load dataset
    train_dataset = pd.read_csv('../../data/csv/{}_{}.csv'.format(exp_name, data_subset["train"]), delimiter=',', header=None, dtype=np.float64).values

    train_min = np.min(train_dataset, axis=0)
    train_max = np.max(train_dataset, axis=0)
    train_dataset = 2.0 * (train_dataset - train_min) / (train_max - train_min) - 1.0

    train_dataset = KoopData(train_dataset, traj_len=traj_len, k=k, d=memory)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = pd.read_csv('../../data/csv/{}_{}.csv'.format(exp_name, data_subset["test"]), delimiter=',', header=None, dtype=np.float64).values
    val_dataset = 2.0 * (val_dataset - train_min) / (train_max - train_min) - 1.0
    val_dataset = KoopData(val_dataset, traj_len=traj_len, k=k, d=memory)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = pd.read_csv('../../data/csv/{}_{}.csv'.format(exp_name, data_subset["test"]), delimiter=',', header=None, dtype=np.float64).values
    test_dataset = 2.0 * (test_dataset - train_min) / (train_max - train_min) - 1.0
    test_dataset = KoopData(test_dataset, traj_len=traj_len, k=k, d=memory)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Dataset is empty")
    
        

    model = Lusch_Model(params, koopman_mode, koopman=koopman, omegas=omegas, d=memory).to(device)
    # load model if applicable
    if load_model is not None:
        checkpoint = torch.load(load_model)
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        for net, net_state in zip(model.auxillary, checkpoint['auxillary_state_dict']):
            net.load_state_dict(net_state)
        print("Loaded model from ", load_model)


    model_path = train_lusch(model, 
                params,
                lambdas, 
                train_loader, 
                val_loader, 
                test_loader, 
                k, 
                1,
                learning_rate, 
                koopman_mode, 
                load=load, #f"../models/{exp_name}_epoch_999.pth", 
                name=koopman_mode+wandb_name,
                use_log=use_log,
                start_epoch=start_epoch)

    pr.disable()

    if not os.path.exists("../../profiler_logs"):
        os.makedirs("../../profiler_logs")
    with open(f"../../profiler_logs/{exp_name}_cprofile.log", "w") as f:
        ps = pstats.Stats(pr, stream=f).sort_stats("tottime")
        ps.print_stats()

    plot_lusch(model, params, train_loader, model_path, name=None)
    wandb.finish()
    print("Finished Experiment")

def sweep_spectral(nc_max=3, nr_max=6,eig_max=6):
    exp_name = "Pendulum" # "DiscreteSpectrumExample", "FluidFlowBox", "FluidFlowOnAttractor", "Pendulum", "Lorenz"
    data_subset = {"train":"train1_x", "test":"test_x"}
    wandb_enabled = True 
    batch_size = 32
    data_subset = {"train":"train1_x", "test":"test_x"}

    koopman_mode = "lusch_params" # "koopman" for precomputed koopman, "lusch_params " for randomly-initialized auxillary network, "hot_start" for hot starting auxillary network, "no_hot_start" for using lusch with koop params, but not fillinmg eigenvalues
    params = importlib.import_module("{}ExperimentBestParams".format(exp_name)).params
    helperfns.set_defaults(params)

    params['wandb_enabled'] = wandb_enabled 
    params['exp_name'] = exp_name
    params['koopman_mode'] = koopman_mode
    params['batch_size'] = batch_size
    params['use_log_real'] = True
    params['num_shifts_middle'] = 3

    # Note did not do 1c, 6r yet
    # eig_reals = [0, 6]
    # eig_complex = [3, 0]
    for i in range(0,nc_max+1):
        for j in range(0,nr_max+1):
    # for i in eig_complex:
    #     for j in eig_reals:
            if i == 0 and j == 0:
                continue
            print(i)
            print(j)
            if 2*i+j > eig_max:
                continue
            exp_params = deepcopy(params)
            exp_params['num_complex_pairs'] = i
            exp_params['num_real'] = j
            # load_path=f"../models/Lorenz_nc{i}_nr{j}_mem1_epoch_299.pth"
            # run_exp(exp_params, load=load_path, start_epoch=299)
            run_exp(exp_params)

def sweep_order(order_max=5):
    exp_name = "Lorenz" # "DiscreteSpectrumExample", "FluidFlowBox", "FluidFlowOnAttractor", "Pendulum", "Lorenz"
    data_subset = {"train":"train1_x", "test":"test_x"}
    wandb_enabled = True 
    batch_size = 32
    data_subset = {"train":"train1_x", "test":"test_x"}

    koopman_mode = "hot_start" # "koopman" for precomputed koopman, "lusch_params " for randomly-initialized auxillary network, "hot_start" for hot starting auxillary network, "no_hot_start" for using lusch with koop params, but not fillinmg eigenvalues
    params = importlib.import_module("{}ExperimentBestParams".format(exp_name)).params
    helperfns.set_defaults(params)

    params['wandb_enabled'] = wandb_enabled 
    params['koopman_mode'] = koopman_mode
    params['batch_size'] = batch_size
    params['use_log_real'] = True
    params['num_shifts_middle'] = 3

    # Note did not do 1c, 6r yet
    # eig_reals = [0, 6]
    # eig_complex = [3, 0]
    for i in range(2,order_max):
            exp_params = deepcopy(params)
            exp_params['exp_name'] = exp_name
            exp_params['koopman_path'] = "../../data/koopmans/{}_{}.mat".format(exp_name+str(int(i)), "observables")
            run_exp(exp_params)
def main():
    exp_name = "Pendulum" # "DiscreteSpectrumExample", "FluidFlowBox", "FluidFlowOnAttractor", "Pendulum", "Lorenz"
    data_subset = {"train":"train1_x", "test":"test_x"}
    wandb_enabled = True 
    batch_size = 128

    koopman_mode = "lusch_params" # "koopman" for precomputed koopman, "lusch_params" for randomly-initialized auxillary network, "hot_start" for hot starting auxillary network, "no_hot_start" for using lusch with koop params, but not fillinmg eigenvalues
    params = importlib.import_module("{}ExperimentBestParams".format(exp_name)).params
    helperfns.set_defaults(params)

    ## Runs one experiment
    params['use_log_real'] = False
    params['wandb_enabled'] = wandb_enabled 
    params['exp_name'] = exp_name
    params['koopman_mode'] = koopman_mode
    params['batch_size'] = batch_size
    params["swap_mode"] = False

    # NOTE: to load model, replace this with path to model (relative) and change start_epoch to that model's epoch
    
    params['load_model']= None
    params['start_epoch'] = 0
    params['num_shifts_middle'] = 10
    params['swap_mode'] = False
    # run_exp(params, data_subset)

    ## Runs multiple experiments trying different no. of eigenvalues
    sweep_spectral()
    # sweep_order()

def main_old():

    pr = cProfile.Profile()
    pr.enable()
    # Define hyperparams
    # exp_name = "DiscreteSpectrumExample" # "DiscreteSpectrumExample", "FluidFlowBox", "FluidFlowOnAttractor", "Pendulum"
    exp_name = "DiscreteSpectrumExample" # "DiscreteSpectrumExample", "FluidFlowBox", "FluidFlowOnAttractor", "Pendulum", "Lorenz"
    data_subset = {"train":"train1_x","val":"val_x", "test":"test_x"}
    wandb_enabled = True 

    koopman_mode = "lusch_params" # "koopman" for precomputed koopman, "lusch_params" for randomly-initialized auxillary network, "hot_start" for hot starting auxillary network, "no_hot_start" for using lusch with koop params, but not fillinmg eigenvalues
    params = importlib.import_module("{}ExperimentBestParams".format(exp_name)).params
    helperfns.set_defaults(params)
    learning_rate = params['learning_rate']
    batch_size = 128
    # batch_size = 64
    traj_len = params['len_time']
    # print(params["num_shifts_middle"])
    k = 20
    # k=params['num_shifts_middle'] # number of middle shifts

    lambdas = {
        "recon": params['recon_lam'],
        "forward": params['recon_lam'],
        "linearity": params['mid_shift_lam'],
        "L1": params["L1_lam"],
        "L2": params["L2_lam"],
        "Linf": params["Linf_lam"]
    }

    # test override, all lambdas 0 but reconstruction
    # lambdas['forward'] = 0.0
    # lambdas['linearity'] = 0.0
    # lambdas['L1'] = 0.0
    # lambdas['L2'] = 0.0
    # lambdas['Linf'] = 0.0

    # Load precomputed koopman operator
    koopman_path = "../../data/koopmans/{}_{}.mat".format(exp_name, "observables")
    koopman, d, omegas = load_koopman(koopman_path)
    koopman = koopman.to(device)
    hidden_size = koopman.shape[0]

    # test

    print("Koopman shape: ", koopman.shape)
    print("d: ", d)
    
    if wandb_enabled:
        wandb.init(project="KoopmanGrowth", 
                config={"learning_rate": learning_rate, 
                    "batch_size": batch_size, 
                    "traj_len": traj_len, 
                    "k - history": k,
                    "reconstruction loss": lambdas['recon'],
                    "forward loss": lambdas['forward'],
                    "linearity loss": lambdas['linearity'],
                    "aux_setting": "No Auxillary Network"})
        

        wandb_name = wandb.run.name
        extra_name = "Precomputed Koopman"
        extra_name = extra_name + "-" if extra_name != "" else extra_name
        wandb.run.name = exp_name + "-" + extra_name + wandb.run.name

    # Load dataset
    train_dataset = pd.read_csv('../../data/csv/{}_{}.csv'.format(exp_name, data_subset["train"]), delimiter=',', header=None, dtype=np.float64).values
    train_dataset = KoopData(train_dataset, traj_len=traj_len, k=k, d=d)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = pd.read_csv('../../data/csv/{}_{}.csv'.format(exp_name, data_subset["val"]), delimiter=',', header=None, dtype=np.float64).values
    val_dataset = KoopData(val_dataset, traj_len=traj_len, k=k, d=d)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = pd.read_csv('../../data/csv/{}_{}.csv'.format(exp_name, data_subset["test"]), delimiter=',', header=None, dtype=np.float64).values
    test_dataset = KoopData(test_dataset, traj_len=traj_len, k=k, d=d)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Dataset is empty")

    # Define model
    params['wandb_enabled'] = wandb_enabled 
    params['exp_name'] = exp_name
    model = Lusch_Model(params, koopman_mode, koopman=koopman, omegas=omegas, d=d).to(device)
    d = 1

    train_lusch(model, 
                params,
                lambdas, 
                train_loader, 
                val_loader, 
                test_loader,
                k, 
                d,
                learning_rate, 
                koopman_mode, 
                load=None, #f"../models/{exp_name}_epoch_999.pth", 
                name=koopman_mode)

    wandb.finish()

    pr.disable()

    if not os.path.exists("../../profiler_logs"):
        os.makedirs("../../profiler_logs")
    
    with open(f"../../profiler_logs/{exp_name}_cprofile.log", "w") as f:
        ps = pstats.Stats(pr, stream=f).sort_stats("tottime")
        ps.print_stats()
    print("Finished training")

if __name__ == "__main__":
    main()