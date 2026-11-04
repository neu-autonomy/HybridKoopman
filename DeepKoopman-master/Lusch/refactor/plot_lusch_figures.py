from matplotlib.colors import LinearSegmentedColormap
import torch
import numpy as np
import wandb
import importlib
import helperfns
from utils import *
from torch.utils.data import DataLoader
import pandas as pd
from training_new import Lusch_Model
from copy import copy
import CalculateFrequency as cf

def display_fig3(model, params):
    encoder = model.encoder.cpu()
    decoder = model.decoder.cpu()
    auxillary = [model.auxillary[i].cpu() for i in range(len(model.auxillary))]  # Move auxillary networks to CPU
    num_complex_pairs = params['num_complex_pairs']
    num_real = params['num_real']

    # set netwokrs to eval
    encoder.eval()
    decoder.eval()
    for aux in auxillary:
        aux.eval()

    # Load data
    ds_example_trajs = pd.read_csv("../postprocessing/DiscreteSpectrumExampleTrajs.csv", delimiter=',', header=None, dtype=np.float64).values
    ds_example_trajs = torch.tensor(ds_example_trajs, dtype=torch.float32)
    ds_example_trajs  = ds_example_trajs.reshape((ds_example_trajs.shape[0],int(ds_example_trajs.shape[1]/2), 2))
    # encode
    with torch.no_grad():
        # compress into batch
        temp = ds_example_trajs.reshape((-1,2))
        encoded_trajs = encoder(temp)

    # We plot some nice trajectories from this system
    # Left side of Figure 3
    start = 0.0
    stop = 1.0
    number_of_lines= 25 
    cm_subsection = np.linspace(start, stop, number_of_lines) 

    colors = plt.get_cmap('jet')(cm_subsection)  # Get the colors from the colormap
    colors = [colors[i] for i in range(len(colors))]  # Convert to list
    
    fig = plt.figure(figsize=(16/3*2, 16/3*2), dpi=200)
    ax = fig.add_subplot(111)
    fig.set_figwidth(16/3*2)
    fig.set_figheight(16/3*2)

    for j in np.arange(ds_example_trajs.shape[0]):
        temp = copy(ds_example_trajs[j,:,:])
        plt.plot(temp[:,0],temp[:,1], color=colors[j])
    xlab = [-.4,0,.4]
    xlabels = ''
    plt.xticks(xlab,xlabels)
    ylab = [-.4, 0, .4]
    plt.yticks(ylab, xlabels)
    plt.ylim([-.4, .55])
    plt.xlim([-.4, .4])
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    plt.savefig('ToyExample.svg', dpi=200, transparent=False)

    trajs_lambdas = np.zeros(ds_example_trajs.shape)

    # print(encoded_trajs.shape)
    omegas_complex, omegas_real = get_omegas_from_auxillary(model, params, encoded_trajs) #comes as list of omegas, where each piece is [num_traj, traj_len]
    # put them together but only keep real parts
    continuous_evals = omegas_complex + omegas_real
    continuous_evals = [np.array(continuous_evals[i].detach()) for i in range(len(continuous_evals))] #convert to numpy array
    # reshape to be num_traj, traj_len, num_complex_pairs
    continuous_evals = np.array(continuous_evals).reshape((ds_example_trajs.shape[0], ds_example_trajs.shape[1], num_real))

    # calculate eval^k for k = 0, 1, ....
    mean_discrete_evals = np.zeros((len(continuous_evals),))
    for k in np.arange(ds_example_trajs.shape[1]):
        for l in np.arange(ds_example_trajs.shape[2]):
            trajs_lambdas[j,k,l] = np.power(mean_discrete_evals[l],k)

    # reshape encoded back to original shape
    encoded_trajs = encoded_trajs.reshape((ds_example_trajs.shape[0], ds_example_trajs.shape[1], 2)).detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_figwidth(16/3*2)
    fig.set_figheight(16/3*2)

    subset = np.arange(0,ds_example_trajs.shape[1],4)

    for j in np.arange(ds_example_trajs.shape[0]):
        scale0 = encoded_trajs[j,0,0]
        scale1 = encoded_trajs[j,0,1]
        plt.plot(np.array(trajs_lambdas[j,:,0]*scale0), np.array(trajs_lambdas[j,:,1]*scale1), 'k', linewidth=2)
        ax.scatter(np.array(encoded_trajs[j,subset,0]),np.array(encoded_trajs[j,subset,1]), color=colors[j])

        
    #plt.axis('equal')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    xlab = [-.4,0,.4]
    xlabels = ''
    plt.xticks(xlab,xlabels)
    ylab = [-.15, 0, .2]
    plt.yticks(ylab, xlabels)
    # plt.ylim([-.4, .55])
    # plt.xlim([-.4, .4])

    plt.savefig('ToyExampleLinear.svg', dpi=200, transparent=False)


def display_fig4(model, params):
    ncoder = model.encoder.cpu()
    decoder = model.decoder.cpu()
    auxillary = [model.auxillary[i].cpu() for i in range(len(model.auxillary))]  # Move auxillary networks to CPU
    num_complex_pairs = params['num_complex_pairs']
    num_real = params['num_real']

    # set netwokrs to eval
    encoder.eval()
    decoder.eval()
    for aux in auxillary:
        aux.eval()

    # Load data
    p_example_trajs = pd.read_csv("../postprocessing/PendulumRings.csv", delimiter=',', header=None, dtype=np.float64).values
    p_example_trajs = torch.tensor(p_example_trajs, dtype=torch.float32)
    rings  = p_example_trajs.reshape((p_example_trajs.shape[0],int(p_example_trajs.shape[1]/2), 2))
    # encode
    with torch.no_grad():
        # compress into batch
        temp = p_example_trajs.reshape((-1,2))
        encoded_trajs = encoder(temp)

    rings_yk = np.zeros(rings.shape)
    rings_yk_transform = np.zeros(rings.shape,dtype=np.complex64)
    rings_recon = np.zeros(rings.shape)
    rings_pred = np.zeros(rings.shape)
    rings_lambdas = np.zeros(rings.shape,dtype=np.complex64)

    for j in np.arange(rings.shape[0]):
        traj = np.squeeze(rings[j,:,:])

        lenTtraj = len(traj)
        
        # apply the network to each trajectory (ring)
        traj_yk, traj_ykplus1, traj_ykplus2, traj_ykplus3, traj_xk_recon, traj_xkplus1, traj_xkplus2, traj_xkplus3 = n.ApplyKoopmanNetOmegas(traj, W, b, params['delta_t'], params['num_real'], params['num_complex_pairs'], params['num_encoder_weights'], params['num_omega_weights'], params['num_decoder_weights'])
        # apply the auxiliary network to each one
        omegas_traj = n.omega_net_apply(traj_yk, W, b, params['num_real'], params['num_complex_pairs'], params['num_omega_weights'])
        
        rings_yk[j,:,:] = traj_yk.copy()
        rings_recon[j,:,:] = traj_xk_recon.copy()
        rings_pred[j,:,:] = traj_xkplus1.copy()
        
        # what's the average omega (real part of eigenvalue) for this trajectory?
        # (note that for this system, we expect this to be constant along trajectories)
        omega_av_traj = np.mean(omegas_traj[0])
        print(omega_av_traj.shape)

        # create 2x2 K matrix with this omega
        L_approx_traj = np.asmatrix([[np.cos(omega_av_traj*deltat), np.sin(omega_av_traj*deltat)], [-np.sin(omega_av_traj*deltat), np.cos(omega_av_traj*deltat)]])
        # calculate the eigenvalues
        # note that this is the discrete case: exp(lambda * deltat) if lambda is the eigenvalue in the continuous case
        evals_net, T_net = np.linalg.eig(L_approx_traj)
        print(evals_net)

        # calculate eval^k for k = 0, 1, ....
        # should match if really have linear system
        for k in np.arange(lenTtraj-1):
            for l in np.arange(len(evals_net)):
                rings_lambdas[j,k,l] = np.power(evals_net[l],k)
        
        # trajectory was mapped into y coordinates
        # now multiply by eigenvector matrix (diagonaliation)
        yk_net_traj_transform = np.asarray(traj_yk*T_net)
        rings_yk_transform[j,:,:] = yk_net_traj_transform.copy()

    colors = [(1, 0, 0), (.5, 0, .5), (0, 0, 1)]
    mymap = LinearSegmentedColormap.from_list('red_purple_blue', colors, N=10)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.set_figwidth(10)
    fig.set_figheight(10)

    for j in np.arange(p_example_trajs.shape[0]):
        # again, stop when reach one period
        inds = np.nonzero(tSpan <= periods[j])
        ind = np.max(inds)+1
        plt.scatter(np.array(encoded_trajs[j,:ind,0]),np.array(encoded_trajs[j,:ind,1]), color=mymap(j))


    xlab = [-.08,0,.08]
    xlabels = ''
    plt.xticks(xlab,xlabels)
    ylab = [-.08,0,.08]
    ylabels = ''
    plt.yticks(ylab,ylabels)

    plt.axis('equal')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    plt.savefig('PendulumLinear.svg', dpi=200, transparent=True)

    fig = plt.figure(figsize=(16/3*2,16/3*2))
    ax = fig.add_subplot(111)

    for j in np.arange(p_example_trajs.shape[0]):
        inds = np.nonzero(tSpan <= periods[j])
        ind = np.max(inds)+1
        subset = np.arange(0,ind,4)

        scale = np.mean(np.abs(rings_yk_transform[j,:,0]))
        
        ind_pos = np.nonzero(p_example_trajs[j,:ind,1] >= 0)[0]
        ind_neg = np.nonzero(p_example_trajs[j,:ind,1] < 0)[0]
        
        plt.plot(np.array(np.real(rings_lambdas[j,:ind,0]*scale)), np.array(np.imag(rings_lambdas[j,:ind,0]*scale)), 'k', linewidth=2)
        plt.plot(np.array(np.real(rings_lambdas[j,:ind,0]*scale)), np.array(np.imag(rings_lambdas[j,:ind,1]*scale)), 'k', linewidth=2)
        ax.scatter(np.array(np.real(rings_yk_transform[j,subset,0])),np.array(np.imag(rings_yk_transform[j,subset,0])), color=mymap(j))


    xlab = [-.05,0,.05]
    xlabels = ''
    plt.xticks(xlab,xlabels)
    ylab = [-.05,0,.05]
    ylabels = ''
    plt.yticks(ylab,ylabels)
    #plt.axis('off')
    #plt.axis('equal')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    plt.savefig('PendulumLinear.svg', dpi=200, transparent=True)

def periodPendulum(theta0, tol, maxN, printFlag=0):
    periodApprox = 2 * np.pi  # n = 0 term
    k = kfn(theta0)
    for n in (np.arange(maxN) + 1):  # n = 1, 2, ..., maxN
        if n < 3:
            nextTerm = periodTerm(n, k)
        else:
            # nextTermOldFormula = periodTerm(n, k)
            nextTerm = periodTermNextStep(n, k, prevTerm)
            # if (np.abs(nextTermOldFormula - nextTerm) > 10**(-10)):
            #    print "discrepancy at n = %f, k = %f: prev. formula said nextTerm = %f, new formula says nextTerm = %f" % (n, k, nextTermOldFormula, nextTerm)
        if nextTerm < 0:
            print("nextTerm < 0: %f at n = %f, k = %f" % (nextTerm, n, k))
        if nextTerm > tol:  # nextTerm gives sense of error (really, lower bound, since all terms are non-neg)
            periodApprox += nextTerm
        else:
            if printFlag:
                print("reached tol (lower bound on error) after n = %d" % (n - 1))
            break
        prevTerm = nextTerm.copy()
    return periodApprox

if __name__ == "__main__":
    exp_name = "DiscreteSpectrumExample"
    data_subset = {"train":"train1_x", "test":"test_x"}
    # wandb_enabled =False 
    epoch = 999
    ckpt = f"../models/{exp_name}_epoch_{epoch}.pth"
    wandb_enabled =False
    use_aux = True
    koopman_mode = "lusch_params"
    params = importlib.import_module("{}ExperimentBestParams".format(exp_name)).params
    helperfns.set_defaults(params)
    learning_rate = params['learning_rate']
    batch_size = 25
    traj_len = params['len_time']
    k = 10
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
    koopman_path = "../../data/koopmans/{}_{}.mat".format(exp_name, "observables") 
    koopman, d, omegas  = load_koopman(koopman_path)
    koopman = koopman.to(device)
    
    hidden_size = koopman.shape[0]
    # Define model
    model = Lusch_Model(params, koopman_mode=koopman_mode, koopman=koopman, omegas=omegas, d=d).to(device)
    d=1

    encoder = model.encoder
    decoder = model.decoder
    auxillary = model.auxillary

    checkpoint = torch.load(ckpt, weights_only=False)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    for i, aux_model in enumerate(auxillary):
        aux_model.load_state_dict(checkpoint['auxillary_state_dict'][i])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    # display_fig3(model, params)
    display_fig4(model, params)

    
    